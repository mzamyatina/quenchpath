"""Per-location CH4-to-CO reaction-network construction and shortest-path
search.

Deliberately pure and iris/grid-free: every function here takes plain
Python floats/dicts/lists (already resolved to one grid location), not
iris Cubes or array+index pairs.
"""

import networkx as nx

# What: {reaction eqn string: [(node_from, node_to), ...]} - one tuple
# per carbon atom conserved, written for the forward (reacs->prods)
# direction. Never cross one tuple's from with another's to.
#
# Why 2 tuples for a reaction with 2 carbon-bearing reactants/products:
# build_unidirect_graph() connects every reactant to every product with
# no notion of which keeps which carbon, so each real pairing needs its
# own tuple to say which one is real.
#
# Why no separate reverse-direction check: a reverse-favoured edge is
# checked against each pair's own reverse (build_unidirect_graph()
# already flips it to (prod, reac)).
#
# What: see prune_to_carbon_subgraph(), which uses this to prune the
# carbon-only subgraph - an edge whose reaction isn't a key here is
# kept unquestioned; one that is a key but not in its list gets
# removed. One known ambiguity - see GitHub issue #1.
EQNS_W_FOUR_C = {
    "CH4+1CH2↔CH3+CH3": [("CH4", "CH3"), ("1CH2", "CH3")],
    "HCO+1CH2↔CH3+CO": [("HCO", "CO"), ("1CH2", "CH3")],
    "HCO+3CH2↔CH3+CO": [("HCO", "CO"), ("3CH2", "CH3")],
    "HCO+CH3↔CH4+CO": [("HCO", "CO"), ("CH3", "CH4")],
    "CH3+H2CO↔HCO+CH4": [("CH3", "CH4"), ("H2CO", "HCO")],
    "HCO+HCO↔H2CO+CO": [("HCO", "CO"), ("HCO", "H2CO")],
    "H2CO+1CH2↔CH3+HCO": [("H2CO", "HCO"), ("1CH2", "CH3")],
    "CH3O+1CH2↔CH3+H2CO": [("CH3O", "H2CO"), ("1CH2", "CH3")],
    "CH3O+3CH2↔CH3+H2CO": [("CH3O", "H2CO"), ("3CH2", "CH3")],
    "CH3O+CH3↔H2CO+CH4": [("CH3O", "H2CO"), ("CH3", "CH4")],
    "CH3O+CH4↔CH3+CH3OH": [("CH3O", "CH3OH"), ("CH4", "CH3")],
    "CH3O+CO↔CH3+CO2": [("CH3O", "CH3"), ("CO", "CO2")],
    "CH3O+HCO↔CH3OH+CO": [("CH3O", "CH3OH"), ("HCO", "CO")],
    "CH3O+H2CO↔CH3OH+HCO": [("CH3O", "CH3OH"), ("H2CO", "HCO")],
    "CH3O+CH3O↔CH3OH+H2CO": [("CH3O", "CH3OH"), ("CH3O", "H2CO")],
    "CH2OH+3CH2↔CH3+H2CO": [("CH2OH", "H2CO"), ("3CH2", "CH3")],
    "CH2OH+CH3↔CH4+H2CO": [("CH2OH", "H2CO"), ("CH3", "CH4")],
    "CH2OH+CH4↔CH3OH+CH3": [
        ("CH2OH", "CH3OH"),
        ("CH4", "CH3"),
    ],
    "CH2OH+HCO↔CH3OH+CO": [("CH2OH", "CH3OH"), ("HCO", "CO")],
    "CH2OH+HCO↔H2CO+H2CO": [("CH2OH", "H2CO"), ("HCO", "H2CO")],
    "CH2OH+H2CO↔CH3OH+HCO": [("CH2OH", "CH3OH"), ("H2CO", "HCO")],
    "CH2OH+CH3O↔CH3OH+H2CO": [("CH2OH", "H2CO"), ("CH3O", "CH3OH")],
    "CH2OH+CH2OH↔CH3OH+H2CO": [("CH2OH", "CH3OH"), ("CH2OH", "H2CO")],
    "CH3OH+1CH2↔CH2OH+CH3": [("CH3OH", "CH2OH"), ("1CH2", "CH3")],
    "CH3OH+3CH2↔CH3+CH2OH": [("CH3OH", "CH2OH"), ("3CH2", "CH3")],
    "CH3OH+3CH2↔CH3+CH3O": [("CH3OH", "CH3O"), ("3CH2", "CH3")],
    "CH3OH+CH3O↔CH3OH+CH2OH": [("CH3OH", "CH2OH"), ("CH3O", "CH3OH")],
    "CO2+3CH2↔H2CO+CO": [("CO2", "CO"), ("3CH2", "H2CO")],
    "HOCN+CH3↔NCO+CH4": [("HOCN", "NCO"), ("CH3", "CH4")],
    "CN+CO2↔NCO+CO": [("CN", "NCO"), ("CO2", "CO")],
    "HNCO+CN↔HCN+NCO": [("HNCO", "NCO"), ("CN", "HCN")],
    "NCO+H2CO↔HNCO+HCO": [("NCO", "HNCO"), ("H2CO", "HCO")],
    "H2CN+CH3↔HCN+CH4": [("H2CN", "HCN"), ("CH3", "CH4")],
    "NCO+HCO↔HNCO+CO": [("NCO", "HNCO"), ("HCO", "CO")],
    "NCO+NCO↔CO+CO+N2": [("NCO", "CO")],
    "CN+CH4↔HCN+CH3": [("CN", "HCN"), ("CH4", "CH3")],
    "NCO+CH4↔HNCO+CH3": [("NCO", "HNCO"), ("CH4", "CH3")],
    "CN+H2CO↔HCN+HCO": [("CN", "HCN"), ("H2CO", "HCO")],
}


def species_order(eqns):
    """Order-preserving list of every species (reactant or product)
    appearing across `eqns`'s "reacs"/"prods". Same across every grid
    location in a run, so callers iterating many locations should
    compute this once and reuse it.

    Parameters
    ----------
    eqns: list[dict]
        One dict per reaction, each with "reacs" (list[str]) and "prods"
        (list[str]).

    Returns
    -------
    list[str]
        Every species name, in first-appearance order, with duplicates
        dropped.
    """
    species = [s for eqn in eqns for s in eqn["reacs"] + eqn["prods"]]
    return list(dict.fromkeys(species))


def build_unidirect_graph(eqns, nodes=None):
    r"""Build the full reaction-network graph for one grid location.

    What: each edge is weighted by its own 'seg_rate_inv' (whichever
    direction k_eq favours):

    .. math::
        \mathrm{seg\_rate\_inv} = \frac{1}{\mathrm{rate}_f}
        \;\text{or}\; \frac{1}{\mathrm{rate}_r}

    Why weight by 1/rate: dijkstra finds the shortest (smallest-weight)
    path, so weighting by 1/rate means it finds the fastest, not
    slowest, path.

    Parameters
    ----------
    eqns: list[dict]
        One dict per reaction, each with "eqn" (str), "reacs" (list[str]),
        "prods" (list[str]), and this location's already-indexed scalar
        "k_eq"/"rate_f"/"rate_r" values (floats, not Cubes/arrays).
    nodes: list[str], optional
        Precomputed species_order(eqns) - pass this in when calling
        repeatedly for the same reaction set (different locations) to
        skip recomputing it every time. Defaults to species_order(eqns).

    Returns
    -------
    networkx.MultiDiGraph
        Weighted 'seg_rate_inv' edges (see above), plus 'eqn' and
        'direction' ("forward"/"reverse" - whether k_eq favoured
        `eqn`'s reacs->prods as written or its reverse) edge
        attributes. Nodes are exactly `nodes` (`species_order(eqns)` if
        not given) - every edge endpoint already comes from some
        reaction's reacs/prods, already covered by `nodes`.
    """
    if nodes is None:
        nodes = species_order(eqns)

    edges_by_key = {}
    for eqn in eqns:
        # What: compute direction/rate once here rather than once per
        # pair below.
        # Why: direction/rate depend only on the reaction, not
        # reac/prod.
        if eqn["k_eq"] >= 1:
            direction, rate = "forward", eqn["rate_f"]
        else:
            direction, rate = "reverse", eqn["rate_r"]
        if rate <= 0:
            # What: skip the edge entirely.
            # Why: a genuinely-zero rate (e.g. float64 underflow at
            # some cold/deep location - see chemistry/formulas.py's own
            # underflow comments) is "no route", not "infinitely fast"
            # - letting 1/rate silently become inf would propagate a
            # bogus near-instant timescale.
            continue
        seg_rate_inv = float(1 / rate)
        for reac in eqn["reacs"]:
            for prod in eqn["prods"]:
                u, v = (reac, prod) if direction == "forward" else (prod, reac)
                # What: dedup keyed on (from, to, eqn) - the edge's
                # real identity (first occurrence wins).
                # Why this key shape: handles two cases -
                # O-3P+1CH2<->CO+H+H's own two "H" products would
                # otherwise duplicate the ('O-3P', 'H') edge; and two
                # DIFFERENT reactions sharing a rate (real case:
                # O-3P+1CH2<->CO+H+H vs O-3P+1CH2<->CO+H2) must still
                # stay separate, so seg_rate_inv can't be part of the
                # key.
                key = (u, v, eqn["eqn"])
                attrs = {"eqn": eqn["eqn"], "seg_rate_inv": seg_rate_inv, "direction": direction}
                edges_by_key.setdefault(key, (u, v, attrs))

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges_by_key.values())

    return G


def build_bidirect_graph(eqns, nodes=None):
    r"""What: like build_unidirect_graph(), but emits BOTH directions of
    every reaction - (reac->prod) weighted 1/rate_f AND (prod->reac)
    weighted 1/rate_r - rather than only whichever k_eq favours.

    Why (GitHub issue #16, for a CO-to-CH4 search specifically):
    build_unidirect_graph() orients each edge by k_eq alone, a purely
    thermodynamic (abundance-independent) quantity - at a real
    CO-favouring location this leaves CO with no forward edge into C-H-O
    space at all (verified empirically - see
    scratch/probe_backbone_orientation.py), even though the disfavoured
    direction's rate_r is very often NOT negligible (deep/hot locations
    sit at local chemical equilibrium, where rate_f == rate_r exactly;
    only near/above the quench level does the disfavoured direction's
    rate genuinely collapse). Weighting the disfavoured direction by its
    own real 1/rate_r (rather than admitting it for free, or not at all)
    lets dijkstra pick it up exactly where it's physically cheap and
    avoid it exactly where it's physically expensive, with no separate
    quench-level detection needed.

    What: NOT a drop-in replacement for build_unidirect_graph() -
    deliberately not used for the CH4-to-CO search direction. Use this
    only for a CO-to-CH4 (or other reverse-direction) search; leave
    CH4-to-CO on build_unidirect_graph().
    quenchpath.config.SpeciesConversion is where this choice is
    actually wired to a given conversion.

    Why not for CH4-to-CO: admitting every reaction's
    locally-disfavoured direction rewrites the *existing*,
    already-validated CH4-to-CO answer at 100% of locations tested (an
    unrelated regression, not a fix).

    Parameters
    ----------
    eqns: list[dict]
        Same shape as build_unidirect_graph()'s own `eqns`.
    nodes: list[str], optional
        Same as build_unidirect_graph()'s own `nodes`.

    Returns
    -------
    networkx.MultiDiGraph
        Both directions of every reaction, each its own 'seg_rate_inv'/
        'eqn'/'direction' edge (see build_unidirect_graph()) - deduped on
        (from, to, eqn, direction) rather than build_unidirect_graph()'s
        (from, to, eqn).

        Why the extra `direction` key: here the forward and reverse
        edges of the same reaction must both survive rather than shadow
        each other.
    """
    if nodes is None:
        nodes = species_order(eqns)

    edges_by_key = {}
    for eqn in eqns:
        for reac in eqn["reacs"]:
            for prod in eqn["prods"]:
                # What: only admit whichever of forward/reverse is
                # actually nonzero.
                # Why: a genuinely-zero rate (e.g. float64 underflow at
                # some cold/deep location) is "no route" in that
                # direction, not "infinitely fast" - letting 1/rate
                # silently become inf would be wrong.
                directions = []
                if eqn["rate_f"] > 0:
                    directions.append(("forward", (reac, prod, float(1 / eqn["rate_f"]))))
                if eqn["rate_r"] > 0:
                    directions.append(("reverse", (prod, reac, float(1 / eqn["rate_r"]))))
                for direction, (u, v, seg_rate_inv) in directions:
                    key = (u, v, eqn["eqn"], direction)
                    attrs = {
                        "eqn": eqn["eqn"],
                        "seg_rate_inv": seg_rate_inv,
                        "direction": direction,
                    }
                    edges_by_key.setdefault(key, (u, v, attrs))

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges_by_key.values())

    return G


def prune_to_carbon_subgraph(G, eqns_w_four_c=EQNS_W_FOUR_C, nodes_with_carbon=None):
    """Subgraph of `G` restricted to carbon-containing species (any node
    name containing "C"), with edges pruned to only those matching a
    known carbon-atom-preserving transformation (see EQNS_W_FOUR_C).

    Parameters
    ----------
    G: networkx.MultiDiGraph
        As returned by build_unidirect_graph().
    eqns_w_four_c: dict, optional
        See EQNS_W_FOUR_C - defaults to it.
    nodes_with_carbon: list[str], optional
        Precomputed carbon-containing subset of `G`'s nodes, for callers
        that already have it (e.g. one restricted further to exclude
        some species). Defaults to computing it directly from
        `G.nodes()`.

        Why this is the only leak-free way to exclude a species from a
        search: `build_unidirect_graph()`'s own `nodes=` does NOT stop
        an excluded species reappearing (it adds every eqn's edges
        regardless of `nodes`, and networkx auto-creates any endpoint
        not already present), but `G.subgraph()` genuinely can't create
        nodes and drops every incident edge of an omitted one.

    Returns
    -------
    networkx.MultiDiGraph
        `G`'s subgraph on `nodes_with_carbon`, with any edge whose
        reaction is an EQNS_W_FOUR_C key removed unless it (or its
        reverse) is one of that key's valid pairs.
    """
    if nodes_with_carbon is None:
        nodes_with_carbon = [node for node in G.nodes() if "C" in node]
    SG = G.subgraph(nodes_with_carbon)
    SG_tidy = nx.MultiDiGraph(SG)  # "unfreeze" the (immutable) subgraph view

    edges_to_remove = []
    # edges() repeats a (from, to) pair once per parallel edge; set()
    # collapses that so each pair is only visited once.
    for edge in set(SG_tidy.edges()):
        for key, attr_dict in SG_tidy.get_edge_data(*edge).items():
            eqn = attr_dict["eqn"]
            if eqn in eqns_w_four_c:
                valid_pairs = eqns_w_four_c[eqn]
                valid_list = valid_pairs + [(to, frm) for frm, to in valid_pairs]
                if edge not in valid_list:
                    edges_to_remove.append((edge, key))
    for edge, key in edges_to_remove:
        SG_tidy.remove_edge(*edge, key=key)

    return SG_tidy


def find_pathway(SG_tidy, source="CH4", target="CO"):
    r"""Find the shortest (fastest-rate) path from `source` to `target`
    through `SG_tidy` - CH4-to-CO by default, but genuinely direction-
    generic (see `source`/`target` below - the CO-to-CH4 search calls
    this same function with them swapped, not a separate function).

    .. math::
        \mathrm{net\_rate\_inv} = \sum_i \mathrm{seg\_rate\_inv}_i

    Path-finding picks the path whose seg_rate_inv values SUM to the
    smallest total - not whose slowest step is fastest (a different
    "bottleneck path" problem dijkstra doesn't solve; see issue #14).
    seg_rate_inv only becomes a per-step "how slow" figure downstream
    (e.g. plot edge coloring).

    What: dijkstra's own path only gives the node sequence, not which
    reaction (of possibly several parallel edges) it used at each hop -
    the reaction actually used is whichever parallel edge has the
    smallest seg_rate_inv.

    Why that's correct: a shortest path can never use a hop's pricier
    parallel edge over a cheaper one - confirmed by networkx's own
    _weight_function: for multigraphs, "the minimum edge weight over
    all parallel edges is returned," i.e. min() below just replays
    dijkstra's own rule.

    Parameters
    ----------
    SG_tidy: networkx.MultiDiGraph
        As returned by prune_to_carbon_subgraph().
    source: str, optional
        Path start species. Defaults to "CH4".
    target: str, optional
        Path end species. Defaults to "CO".

    Returns
    -------
    (net_rate_inv, max_rate_inv, nodes, segments):
    (float, float, list[str], list[dict])
        net_rate_inv [cm3 s] is dijkstra's own reported total (summed
        seg_rate_inv across the path - see above); max_rate_inv [cm3 s]
        is the slowest (bottleneck) segment's own seg_rate_inv along
        that same path - the "bottleneck path" quantity dijkstra itself
        doesn't optimize for (see issue #14), computed here anyway since
        it's one max() over segments dijkstra already found (both are in
        cm3 s since rate_f/rate_r, and so seg_rate_inv = 1/rate, are in
        cm-3 s-1 - see quenchpath.stages.stage_04_rates). `nodes` is the
        dijkstra path itself, `source` to `target` (e.g. ["CH4", "CH3",
        ..., "CO"]). `segments` is one flat dict per edge on the path -
        "eqn", "seg_rate_inv" [cm3 s] (1/rate), and "direction"
        ("forward"/"reverse" - whether k_eq favoured `eqn`'s
        reacs->prods as written or its reverse) - in the same order as
        `nodes`, so segments[i] is the (nodes[i], nodes[i+1]) hop; not
        itself repeating those two species, already available from
        `nodes` at that same index.

    Raises
    ------
    networkx.NetworkXNoPath
        If no path exists between `source` and `target`.
    """
    net_rate_inv, nodes = nx.single_source_dijkstra(
        SG_tidy, source=source, target=target, weight="seg_rate_inv"
    )

    segments = []
    for species_from, species_to in zip(nodes, nodes[1:]):
        # What: seg_rate_inv first - it's the min() sort key.
        # Why eqn/direction are included too: they just ride along for
        # the final segment's convenience. min() replays networkx's own
        # multigraph weight rule (see docstring).
        seg_rate_inv, eqn, direction = min(
            (
                (attr_dict["seg_rate_inv"], attr_dict["eqn"], attr_dict["direction"])
                for attr_dict in SG_tidy.get_edge_data(species_from, species_to).values()
            ),
            key=lambda candidate: candidate[0],
        )
        segments.append({"eqn": eqn, "seg_rate_inv": seg_rate_inv, "direction": direction})
    max_rate_inv = max(seg["seg_rate_inv"] for seg in segments)
    return net_rate_inv, max_rate_inv, nodes, segments
