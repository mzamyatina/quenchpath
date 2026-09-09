"""Regression tests for quenchpath.graph.shortest_path. These functions
are pure (plain floats/dicts/networkx graphs, no iris/grid), so this is
synthetic-data-only - no real NetCDF files needed.
"""

import networkx as nx
import pytest

from quenchpath.graph.shortest_path import (
    build_bidirect_graph,
    build_unidirect_graph,
    find_pathway,
    prune_to_carbon_subgraph,
)


def test_build_unidirect_graph_picks_favoured_direction_by_k_eq():
    # k_eq >= 1 favours forward (A->B via rate_f); k_eq < 1 favours
    # reverse (B->A via rate_r).
    eqns = [
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 10.0, "rate_r": 5.0},
        {"eqn": "C↔D", "reacs": ["C"], "prods": ["D"], "k_eq": 0.5, "rate_f": 10.0, "rate_r": 5.0},
    ]
    G = build_unidirect_graph(eqns)

    assert set(G.nodes()) == {"A", "B", "C", "D"}
    assert G.has_edge("A", "B")
    assert not G.has_edge("B", "A")
    assert G.get_edge_data("A", "B")[0]["seg_rate_inv"] == pytest.approx(1 / 10.0)
    assert G.get_edge_data("A", "B")[0]["eqn"] == "A↔B"
    assert G.get_edge_data("A", "B")[0]["direction"] == "forward"

    assert G.has_edge("D", "C")
    assert not G.has_edge("C", "D")
    assert G.get_edge_data("D", "C")[0]["seg_rate_inv"] == pytest.approx(1 / 5.0)
    assert G.get_edge_data("D", "C")[0]["direction"] == "reverse"


def test_build_unidirect_graph_skips_a_genuinely_zero_rate_edge():
    # A zero rate (e.g. float64 underflow) is "no route" in the favoured
    # direction, not "infinitely fast" - the edge must be entirely absent,
    # not present with seg_rate_inv=inf.
    eqns = [
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 0.0, "rate_r": 5.0},
    ]
    G = build_unidirect_graph(eqns)
    assert not G.has_edge("A", "B")
    assert G.number_of_edges() == 0
    assert set(G.nodes()) == {"A", "B"}  # nodes still present, just no edge


def test_build_unidirect_graph_handles_multi_reactant_multi_product():
    # The original script special-cased len(reacs)==1 and len(prods)==1
    # as apparently-distinct branches from the general reacs x prods
    # double loop - all three are mathematically identical (verified via
    # this test: a 2-reactant, 2-product reaction produces all 4 edges).
    eqns = [
        {
            "eqn": "A+B↔C+D",
            "reacs": ["A", "B"],
            "prods": ["C", "D"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_unidirect_graph(eqns)
    expected_edges = {("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")}
    assert set(G.edges()) == expected_edges


def test_build_unidirect_graph_deduplicates_repeated_edges():
    # "Remove duplicate edges caused by reactions like O-3P+1CH2<->CO+H+H,
    # where an ('O-3P', 'H') edge would otherwise be created twice" - see
    # the original script's comment, carried over verbatim.
    eqns = [
        {
            "eqn": "A+B↔C+C",
            "reacs": ["A", "B"],
            "prods": ["C", "C"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_unidirect_graph(eqns)
    # A->C and B->C would each be produced twice (once per "C" in prods)
    # without deduplication.
    assert G.number_of_edges() == 2


def test_build_unidirect_graph_keeps_different_reactions_sharing_a_rate():
    # Two DIFFERENT reactions connecting the same species pair that
    # happen to share the exact same rate (a real case in this
    # project's data: O-3P+1CH2<->CO+H+H and O-3P+1CH2<->CO+H2) must
    # stay two separate parallel edges - deduplication must key on
    # (from, to, eqn), not on (from, to, seg_rate_inv) or the full
    # tuple's rate value.
    eqns = [
        {"eqn": "A↔C", "reacs": ["A"], "prods": ["C"], "k_eq": 2.0, "rate_f": 1.0, "rate_r": 1.0},
        {"eqn": "B↔C", "reacs": ["B"], "prods": ["C"], "k_eq": 2.0, "rate_f": 1.0, "rate_r": 1.0},
    ]
    G = build_unidirect_graph(eqns)
    assert G.number_of_edges() == 2
    assert {d["eqn"] for _, _, d in G.edges(data=True)} == {"A↔C", "B↔C"}


def test_build_unidirect_graph_first_occurrence_wins_on_true_duplicate_reaction():
    # If the same eqn string legitimately appeared twice for the same
    # (from, to) pair with different rates (not expected in real data -
    # see the code review that added this test - but not guarded against
    # upstream either), the first occurrence must win deterministically,
    # matching this project's established convention (see
    # quenchpath.chemistry.polynomials.parse_nasa_polynomials()).
    eqns = [
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 1.0, "rate_r": 1.0},
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 999.0, "rate_r": 1.0},
    ]
    G = build_unidirect_graph(eqns)
    assert G.number_of_edges() == 1
    assert G.get_edge_data("A", "B")[0]["seg_rate_inv"] == pytest.approx(1 / 1.0)


def test_build_bidirect_graph_emits_both_directions():
    # Unlike build_unidirect_graph() (only whichever direction k_eq favours),
    # build_bidirect_graph() keeps both - A->B weighted 1/rate_f
    # AND B->A weighted 1/rate_r, regardless of which one k_eq favours.
    eqns = [
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 10.0, "rate_r": 5.0},
    ]
    G = build_bidirect_graph(eqns)

    assert G.has_edge("A", "B")
    assert G.get_edge_data("A", "B")[0]["seg_rate_inv"] == pytest.approx(1 / 10.0)
    assert G.get_edge_data("A", "B")[0]["direction"] == "forward"

    assert G.has_edge("B", "A")
    assert G.get_edge_data("B", "A")[0]["seg_rate_inv"] == pytest.approx(1 / 5.0)
    assert G.get_edge_data("B", "A")[0]["direction"] == "reverse"


def test_build_bidirect_graph_matches_build_unidirect_graph_on_the_favoured_edge():
    # The favoured-direction edge itself (weight, eqn, direction) must
    # be identical between the two builders - build_bidirect_graph()
    # only ever ADDS the disfavoured edge, never changes the favoured one.
    eqns = [
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 10.0, "rate_r": 5.0},
        {"eqn": "C↔D", "reacs": ["C"], "prods": ["D"], "k_eq": 0.5, "rate_f": 10.0, "rate_r": 5.0},
    ]
    G_uni = build_unidirect_graph(eqns)
    G_bidir = build_bidirect_graph(eqns)

    assert G_uni.get_edge_data("A", "B")[0] == G_bidir.get_edge_data("A", "B")[0]
    assert G_uni.get_edge_data("D", "C")[0] == G_bidir.get_edge_data("D", "C")[0]


def test_build_bidirect_graph_skips_a_genuinely_zero_rate_direction():
    # Same reasoning as test_build_unidirect_graph_skips_a_genuinely_zero_rate_edge()
    # - only the nonzero direction should be admitted, the zero one must
    # be entirely absent, not present with seg_rate_inv=inf.
    eqns = [
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 10.0, "rate_r": 0.0},
    ]
    G = build_bidirect_graph(eqns)
    assert G.has_edge("A", "B")
    assert not G.has_edge("B", "A")
    assert G.number_of_edges() == 1


def test_build_bidirect_graph_deduplicates_repeated_edges_per_direction():
    # Same "A+B<->C+C" case as
    # test_build_unidirect_graph_deduplicates_repeated_edges() - each
    # direction dedupes on its own (from, to, eqn, direction) key, so
    # forward and reverse don't collide with each other, but a repeated
    # product ("C" twice) still collapses within one direction.
    eqns = [
        {
            "eqn": "A+B↔C+C",
            "reacs": ["A", "B"],
            "prods": ["C", "C"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_bidirect_graph(eqns)
    # A->C, B->C (forward) and C->A, C->B (reverse) - 4 edges, not 8.
    assert G.number_of_edges() == 4
    assert set(G.edges()) == {("A", "C"), ("B", "C"), ("C", "A"), ("C", "B")}


def test_find_pathway_needs_bidirect_graph_to_reverse_a_favoured_chain():
    # A->B->C, both reactions favouring forward (like a real CO-favouring
    # location's CO<->HCO - see GitHub issue #16/scratch/probe_backbone_
    # orientation.py): build_unidirect_graph() alone has no C->A route at all,
    # matching CO-to-CH4's real no-path failure mode; build_bidirect_graph()
    # recovers it via the disfavoured (1/rate_r-weighted) edges.
    eqns = [
        {"eqn": "A↔B", "reacs": ["A"], "prods": ["B"], "k_eq": 2.0, "rate_f": 1.0, "rate_r": 1.0},
        {"eqn": "B↔C", "reacs": ["B"], "prods": ["C"], "k_eq": 2.0, "rate_f": 1.0, "rate_r": 1.0},
    ]
    G_uni = build_unidirect_graph(eqns)
    with pytest.raises(nx.NetworkXNoPath):
        find_pathway(G_uni, source="C", target="A")

    G_bidir = build_bidirect_graph(eqns)
    net_rate_inv, max_rate_inv, nodes, segments = find_pathway(G_bidir, source="C", target="A")
    assert nodes == ["C", "B", "A"]
    assert [seg["direction"] for seg in segments] == ["reverse", "reverse"]


def test_prune_to_carbon_subgraph_keeps_unknown_reactions_unquestioned():
    eqns = [
        {
            "eqn": "CH4+X↔CH3+X",
            "reacs": ["CH4", "X"],
            "prods": ["CH3", "X"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_unidirect_graph(eqns)
    SG_tidy = prune_to_carbon_subgraph(G, eqns_w_four_c={})
    # "X" has no carbon, so only the CH4-CH3 edge remains in the subgraph.
    assert set(SG_tidy.nodes()) == {"CH4", "CH3"}
    assert SG_tidy.has_edge("CH4", "CH3")


def test_prune_to_carbon_subgraph_removes_invalid_carbon_routes():
    # Two carbon-bearing reactants (CH4, CH3O), two carbon-bearing
    # products (CH3, CH2O) - produces all 4 reac-x-prod edges, but only
    # (CH4, CH3) is declared a valid carbon-conserving route for this
    # reaction; the other 3 (CH4,CH2O)/(CH3O,CH3)/(CH3O,CH2O) must be
    # pruned even though they're chemically-plausible-looking edges.
    eqns = [
        {
            "eqn": "CH4+CH3O↔CH3+CH2O",
            "reacs": ["CH4", "CH3O"],
            "prods": ["CH3", "CH2O"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_unidirect_graph(eqns)
    assert G.number_of_edges() == 4

    eqns_w_four_c = {"CH4+CH3O↔CH3+CH2O": [("CH4", "CH3")]}
    SG_tidy = prune_to_carbon_subgraph(G, eqns_w_four_c)
    assert set(SG_tidy.edges()) == {("CH4", "CH3")}


def test_prune_to_carbon_subgraph_accepts_reversed_valid_routes():
    # A location where the reverse direction was favoured still counts as
    # valid if the *forward* (from, to) pair is listed - see
    # prune_to_carbon_subgraph()'s "valid_list ... + reversed" handling.
    eqns = [
        {
            "eqn": "CH4↔CH3",
            "reacs": ["CH4"],
            "prods": ["CH3"],
            "k_eq": 0.1,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_unidirect_graph(eqns)
    assert G.has_edge("CH3", "CH4")  # reverse favoured (k_eq < 1)

    eqns_w_four_c = {"CH4↔CH3": [("CH4", "CH3")]}  # only forward direction listed
    SG_tidy = prune_to_carbon_subgraph(G, eqns_w_four_c)
    assert SG_tidy.has_edge("CH3", "CH4")  # kept - reverse of a valid route


def test_build_unidirect_graph_nodes_arg_does_not_restrict_edges():
    # nodes= is a seeding/ordering hint only - build_unidirect_graph() adds
    # every eqn's edges regardless of it, and add_edges_from() auto-
    # creates any endpoint not already present. So passing a restricted
    # `nodes=` is NOT how to exclude a species (a real leakage risk,
    # verified against the actual Venot2019 network before this was
    # relied on) - the real exclusion hook is prune_to_carbon_subgraph()'s
    # own `nodes_with_carbon=`, see test below.
    eqns = [
        {
            "eqn": "NCO+CH4↔HNCO+CH3",
            "reacs": ["NCO", "CH4"],
            "prods": ["HNCO", "CH3"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_unidirect_graph(eqns, nodes=["CH4", "CH3"])
    assert "NCO" in G.nodes()
    assert "HNCO" in G.nodes()


def test_prune_to_carbon_subgraph_node_restriction_equals_post_hoc_node_removal():
    # The leak-free way to run a pathway search excluding N-carbon
    # species: restrict prune_to_carbon_subgraph()'s own
    # nodes_with_carbon=, not build_unidirect_graph()'s nodes= (see the test
    # above). Equivalent to building on the full network and then
    # removing the excluded nodes afterwards (nx.utils.graphs_equal) -
    # confirms prune_to_carbon_subgraph() has no endpoint-independent
    # edge logic that could make the two diverge.
    eqns = [
        {
            "eqn": "NCO+CH4↔HNCO+CH3",
            "reacs": ["NCO", "CH4"],
            "prods": ["HNCO", "CH3"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
        {
            "eqn": "CH3+H2CO↔HCO+CH4",
            "reacs": ["CH3", "H2CO"],
            "prods": ["HCO", "CH4"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    eqns_w_four_c = {
        "NCO+CH4↔HNCO+CH3": [("NCO", "HNCO"), ("CH4", "CH3")],
        "CH3+H2CO↔HCO+CH4": [("CH3", "CH4"), ("H2CO", "HCO")],
    }
    exclude = frozenset({"NCO", "HNCO"})
    G = build_unidirect_graph(eqns)
    nodes = list(G.nodes())

    restricted = prune_to_carbon_subgraph(
        G,
        eqns_w_four_c,
        nodes_with_carbon=[n for n in nodes if "C" in n and n not in exclude],
    )

    removed = nx.MultiDiGraph(G)
    removed.remove_nodes_from(exclude)
    removed = prune_to_carbon_subgraph(
        removed, eqns_w_four_c, nodes_with_carbon=[n for n in removed.nodes() if "C" in n]
    )

    assert nx.utils.graphs_equal(restricted, removed)
    assert "NCO" not in restricted.nodes()
    assert "HNCO" not in restricted.nodes()
    # The surviving C-H-O->C-H-O edge from the 2nd (N-free) reaction
    # must still be there - excluding N-carbon nodes doesn't remove
    # reactions that never involved one.
    assert restricted.has_edge("CH3", "CH4") or restricted.has_edge("CH4", "CH3")


def test_find_pathway_simple_chain():
    G = nx.MultiDiGraph()
    G.add_edge("CH4", "X", seg_rate_inv=1.0, eqn="CH4->X", direction="forward")
    G.add_edge("X", "CO", seg_rate_inv=2.0, eqn="X->CO", direction="reverse")
    net_rate_inv, max_rate_inv, nodes, segments = find_pathway(G)
    assert nodes == ["CH4", "X", "CO"]
    eqns_in_path = [s["eqn"] for s in segments]
    assert eqns_in_path == ["CH4->X", "X->CO"]
    assert [s["direction"] for s in segments] == ["forward", "reverse"]
    assert net_rate_inv == pytest.approx(1.0 + 2.0)  # dijkstra's own summed seg_rate_inv
    assert max_rate_inv == pytest.approx(2.0)  # slowest (bottleneck) segment


def test_find_pathway_source_target_swap_walks_the_same_graph_the_other_way():
    # GitHub issue #15 (CO-to-CH4): no build_unidirect_graph() change needed -
    # find_pathway() already takes source/target, and dijkstra on a
    # directed graph only ever traverses edges in their stored direction,
    # so a source="CO"/target="CH4" search on the exact same graph just
    # finds CO's own shortest route to CH4, not CH4's reversed. Verified
    # empirically on real per-planet data before this was relied on (see
    # SpeciesConversion's own docstring) - this pins down the small-graph
    # case algebraically.
    G = nx.MultiDiGraph()
    G.add_edge("CH4", "X", seg_rate_inv=1.0, eqn="CH4->X", direction="forward")
    G.add_edge("X", "CO", seg_rate_inv=2.0, eqn="X->CO", direction="reverse")
    # A second, cheaper CO->CH4 route that ISN'T just the first path's
    # own reverse - proves this doesn't secretly reuse/reverse the
    # CH4->CO node list, it genuinely reruns dijkstra the other way.
    G.add_edge("CO", "Y", seg_rate_inv=0.1, eqn="CO->Y", direction="forward")
    G.add_edge("Y", "CH4", seg_rate_inv=0.1, eqn="Y->CH4", direction="forward")

    net_rate_inv, _max_rate_inv, nodes, segments = find_pathway(G, source="CO", target="CH4")

    assert nodes == ["CO", "Y", "CH4"]
    assert [s["eqn"] for s in segments] == ["CO->Y", "Y->CH4"]
    assert net_rate_inv == pytest.approx(0.1 + 0.1)


def test_find_pathway_reports_reverse_direction():
    # X↔CH4 (reacs=[X], prods=[CH4]) favours reverse (k_eq < 1), so the
    # CH4->X edge it contributes is the *reverse* of how the reaction is
    # written - segments must report "reverse" for that hop, not just
    # always "forward".
    eqns = [
        {
            "eqn": "X↔CH4",
            "reacs": ["X"],
            "prods": ["CH4"],
            "k_eq": 0.1,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
        {"eqn": "X↔CO", "reacs": ["X"], "prods": ["CO"], "k_eq": 2.0, "rate_f": 1.0, "rate_r": 1.0},
    ]
    G = build_unidirect_graph(eqns)
    net_rate_inv, _max_rate_inv, nodes, segments = find_pathway(G)
    hops = list(zip(nodes, nodes[1:]))
    assert [(*hop, s["direction"]) for hop, s in zip(hops, segments)] == [
        ("CH4", "X", "reverse"),
        ("X", "CO", "forward"),
    ]
    assert net_rate_inv == pytest.approx(1.0 + 1.0)  # both hops' seg_rate_inv is 1/1.0


def test_find_pathway_no_path_raises():
    G = nx.MultiDiGraph()
    G.add_node("CH4")
    G.add_node("CO")  # no edge connecting them
    with pytest.raises(nx.NetworkXNoPath):
        find_pathway(G)


def test_find_pathway_picks_reaction_among_parallel_edges():
    # CH4->X has two parallel edges (competing reactions) with different
    # rates - dijkstra's own path never uses a hop's more expensive
    # parallel edge (fast_reaction's seg_rate_inv=1.0 beats
    # slow_reaction's 5.0), so the reported segment must be
    # "fast_reaction", not just whichever parallel edge happens to be
    # iterated first.
    G = nx.MultiDiGraph()
    G.add_edge("CH4", "X", seg_rate_inv=1.0, eqn="fast_reaction", direction="forward")
    G.add_edge("CH4", "X", seg_rate_inv=5.0, eqn="slow_reaction", direction="forward")
    G.add_edge("X", "CO", seg_rate_inv=2.0, eqn="X->CO", direction="forward")
    net_rate_inv, _max_rate_inv, _nodes, segments = find_pathway(G)
    eqns_in_path = [s["eqn"] for s in segments]
    assert eqns_in_path == ["fast_reaction", "X->CO"]
    assert net_rate_inv == pytest.approx(
        1.0 + 2.0
    )  # via fast_reaction, not slow_reaction's 5.0+2.0


def test_find_pathway_picks_reaction_among_parallel_edges_in_the_swapped_direction_too():
    # Same question as test_find_pathway_picks_reaction_among_parallel_
    # edges() above, but walked source="CO"/target="CH4" (the CO_CH4
    # search's own direction) through a real build_bidirect_graph()
    # output, not a hand-built MultiDiGraph - CO_CH4's graph has
    # parallel edges far more often than CH4_CO's (every reaction
    # contributes both directions, not just whichever k_eq favours), so
    # this is the direction/graph combination most likely to exercise
    # min()'s parallel-edge selection for real. One reaction's
    # DISFAVOURED (reverse-rate) contribution competes against a second
    # reaction's favoured one on the same CO->X hop - the cheap one
    # (X_fast's forward, seg_rate_inv=0.5) must still win over the
    # expensive one (CO_slow's reverse, seg_rate_inv=5.0).
    eqns = [
        {
            "eqn": "X<->CO_slow",
            "reacs": ["X"],
            "prods": ["CO"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 0.2,
        },
        {
            "eqn": "CO<->X_fast",
            "reacs": ["CO"],
            "prods": ["X"],
            "k_eq": 2.0,
            "rate_f": 2.0,
            "rate_r": 1.0,
        },
        {
            "eqn": "X<->CH4",
            "reacs": ["X"],
            "prods": ["CH4"],
            "k_eq": 2.0,
            "rate_f": 1.0,
            "rate_r": 1.0,
        },
    ]
    G = build_bidirect_graph(eqns)
    net_rate_inv, _max_rate_inv, nodes, segments = find_pathway(G, source="CO", target="CH4")
    eqns_in_path = [s["eqn"] for s in segments]
    assert nodes == ["CO", "X", "CH4"]
    assert eqns_in_path == ["CO<->X_fast", "X<->CH4"]
    assert net_rate_inv == pytest.approx(0.5 + 1.0)  # via X_fast, not CO_slow's 5.0+1.0
