# Choosing a uni-directional vs. bidirectional pathway-search graph

A playbook for deciding whether `quenchpath.graph.shortest_path.build_unidirect_graph()`
or `build_bidirect_graph()` is the right graph to search for a given
species conversion (`quenchpath.config.SpeciesConversion`), on a
system you don't yet have priors about.

- `build_unidirect_graph()` - uni-directional: each reaction
  contributes one edge, oriented by that reaction's own local `k_eq(T)`.
- `build_bidirect_graph()` - admits both directions of every reaction,
  each weighted by its own real local rate.
- Written after working this out for CH4<->CO on hot Jupiters (GitHub
  issue #16) - that investigation is the worked example throughout.

**Why three tiers, not one:** a graph construction "working" (finds a
route, no errors, a plausible-looking success rate) is necessary but
not sufficient for being the methodologically right choice.

- Cautionary tale from this codebase: swapping source/target on
  `build_unidirect_graph()`'s *unmodified* output finds a CO-to-CH4
  route at ~99-100% of locations - by a pure success-rate reading,
  that's a working uni-directional search, done.
- Why it's wrong: at a real CO-favouring location, that "route" went
  through a nitrogen-bearing species purely because the direct
  CO<->HCO edge didn't exist at all in the CO-to-CH4 direction, not
  because nitrogen chemistry was genuinely fastest. The search
  "worked" and answered a different, unphysical question.
- Each tier below catches a failure mode the tier before it can't see.

## Tier 1: does it run at all (functional)

1. **Get real data on the grid.** Run the pipeline for both
   experiments (`quenchpath convert` -> `calc-rates` -> `calc-paths-serial`/
   `-parallel` -> `merge-paths` -> `calc-taus`, or `run-pipeline-serial`/
   `-parallel`) for `exp=kin` (real number densities/rates) and
   `exp=equ` (equilibrium-chemistry gradients, needed later for
   `tau_adv_qn` if you get to Tier 3). Nothing below can start before
   this.
2. **Build physical intuition for which direction should dominate.**
   Look at the raw T(p) structure
   (`quenchpath.io.load_converted()`'s `air_temperature`/
   `air_pressure`) at a few representative columns - deep/hot,
   shallow/cool, dayside, nightside.
   - Why this works: equilibrium shifts toward the high-T product at
     high T, the low-T product at low T, so which direction the system
     is "trying to relax toward" is usually visually obvious from the
     T range alone.
   - Why this is scaffolding, not evidence: gives a hypothesis, not an
     answer - per the cautionary tale above, "a route was found" won't
     be enough to confirm it either. Treat everything through Tier 1
     as scaffolding, not evidence.

## Tier 2: is what it finds physically real (local, methodological)

Raw success/failure counts can't tell "found a real route" apart from
"found *a* route, through whatever happened to be graph-connected."
Why this tier exists: that distinction is the entire content of it -
every check below asks not just "did it find something" but "is what
it found the thing it's claiming to be."

3. **Probe the reaction backbone directly, before trusting any
   aggregate number.** Use `build_unidirect_graph()` at a handful of
   representative locations (see `scratch/probe_backbone_orientation.py`
   for the pattern this repo used): pick a location, print whether
   each hop of the known reduction backbone exists in the graph and
   which way it's oriented, edge by edge - not just whether *some*
   path connects the endpoints.
   - Why this matters: a backbone that's fully connected one way but
     missing a link the other way (issue #16 found `CO<->HCO` oriented
     `HCO->CO` only, never the reverse, at a CO-favoured deep location)
     tells you a uni graph's "success" in the reverse direction there
     can only be an artifact - it has to be routing around the gap
     somehow, and you need to see how.
4. **Run the search at scale, and check the routes found are
   elementally real, not just present.** Run `calc-paths-serial`/
   `-parallel` (uni: `build_unidirect_graph()` + `prune_to_carbon_subgraph()` +
   `find_pathway()`) for both directions, then `merge-paths`. Two
   separate checks on `quenchpath.io.load_merged_paths()`'s
   `path_index_cube`/`path_index_map`, not one:
   - **Functional**: did a route exist at all (no failure/masked cell)?
   - **Methodological**: is the route a genuine same-element-family
     chain, or does it lean on an unrelated element's chemistry as a
     connectivity escape hatch (here, nitrogen - the network's
     N-bearing carbon carriers, e.g. HCN/NCO/HNCO;
     `prune_to_carbon_subgraph()`/`EQNS_W_FOUR_C` are what actually
     enforce carbon-atom conservation hop by hop, which is what makes
     this distinction possible in the first place)? And is the escape
     hatch, if used at all, rare and only where it's genuinely fastest
     - or is it present at nearly every location, the signature of the
     tier-0 cautionary tale repeating itself?
   - **Finding**: honest (elementally-checked) numbers were stark -
     CH4->CO uni succeeds cleanly, CHO-only, at near 100% of
     locations. CO->CH4 uni only found a genuine CHO-only route at
     0-6.6% of locations on two test planets - the naive ~99-100%
     "success" from the intro was the unphysical nitrogen-escape-hatch
     version of the same search, caught only by adding the elemental
     check.
5. **Where the uni graph fails the elemental check, try bidirectional
   and re-check the same two things.** Rerun the failing direction
   with `build_bidirect_graph()`.
   - Success rate alone jumping to ~100% is not the bar - check that
     the escape-hatch element is now used only where it's genuinely
     fastest (a few percent), not as a forced fallback, the same way
     step 4 did.
6. **Check bidirectional isn't secretly "more correct" for the other
   direction too.** Try `build_bidirect_graph()` on the direction that
   already passed step 4 uni.
   - If it changes the existing answer *everywhere* (not just filling
     the gaps step 5 found), that's not an improvement - it means it's
     silently admitting thermodynamically-disfavoured routes that
     aren't physically real there. Reject it for that direction, keep
     uni. (Tried and rejected here - see `SpeciesConversion.CH4_CO`'s
     own docstring.)

**Where this leaves you:** a graph choice per direction that is not
just functionally working (Tier 1) but locally, checkably physical
(Tier 2) - every individual route inspected is a real elemental chain,
not a connectivity artifact. That's a substantially stronger claim
than "it ran without errors," and for a lot of purposes it's enough.
What it still hasn't done is ask whether the *premise* behind treating
the two directions asymmetrically - one direction net-favoured almost
everywhere, the other not - is actually true across the whole domain,
or just true at the handful of locations you happened to check.

## Tier 3: is the underlying premise right, everywhere (global, quantitative)

Tier 2 checks are local and per-route: real chain vs. artifact, at the
locations you looked at. None of them ask "which direction is net
downhill" as a domain-wide, quantitative question - that's assumed
from step 2's intuition and never independently re-examined.

Why this tier exists: to close exactly that gap.

7. **Quantify the net system-level direction directly.**
   Why not just use the per-edge `k_eq` every earlier tier relies on:
   it's a *reaction's own* thermodynamic favourability - not the same
   thing as which way the *whole system* is being driven at a
   location, which depends on the real (disequilibrium) composition,
   not just local T (a path can be uphill overall while individual
   steps along it are each locally favoured - these are genuinely
   different quantities, not two views of the same one).
   - Build that separately: the Zahnle & Marley (2014)-style
     K_test/K_eq disequilibrium quotient for the net reaction (here,
     `quenchpath.analysis.k_test_over_k_eq.ch4_co_k_test_over_k_eq()`
     for CH4 + H2O <-> CO + 3H2).
   - Verify it first against a location known to be at real chemical
     equilibrium (ratio should come out ~1 there, not some arbitrary
     number) before trusting it anywhere else.
8. **Find precisely where that assumption is weakest.** "Quenched" is
   conventionally not a single crossing but the timescale ratio being
   >>1 or <<1 - the transition itself is a region, not a level.
   `quenchpath.analysis.quench_region.quench_region()` (built on
   `tau_adv_qn`, the real local composition-gradient-based timescale,
   not a coarse spatial mean) gives a continuous version of that
   region.
   - Cross-reference it against step 7's sign field to see how much of
     the domain - and exactly where - the step-2 assumption actually
     breaks down.
   - **Finding**: essentially nowhere on one test planet, up to ~13%
     of the at/below-quench volume on another, always concentrated
     within a few e-foldings of each column's own quench transition,
     never spread generally through the "quenched" volume.
9. **Test whether that breakdown actually touches the searches.** The
   step that closes the loop back to Tier 2. At the specific locations
   step 8 flagged, redo the tier-2 checks there specifically - real
   pathway-search output (`path_index_cube`/`path_index_map` again),
   route-failure rate, escape-hatch usage, pathway-choice diversity -
   against everywhere else.
   - If neither search's behaviour tracks the sign flip, that's
     direct, positive evidence the Tier 2 decision holds up even where
     Tier 3's own premise gets shakiest - not just an assumption
     you're hoping holds.
   - **Finding**: neither search's behaviour moved at all - checked on
     5 planets, including the two with the largest quench-region
     signal.

## Bottom line

- **Tier 1** tells you the mechanics run.
- **Tier 2** tells you the routes found are real, not artifacts - and
  is not optional if you want to claim anything physical, given how
  cleanly a purely functional check can be fooled (the intro's example
  is a real result from this exact codebase, not a hypothetical).
- **Tier 3** is the only layer that tests the asymmetric-treatment
  premise itself, quantitatively, across the whole domain rather than
  at the handful of locations Tier 2 happened to probe - do it when
  the claim needs to survive scrutiny beyond "we didn't see anything
  wrong locally," which is a real bar for a methods section and a
  reasonable one to hold yourself to even without one.
