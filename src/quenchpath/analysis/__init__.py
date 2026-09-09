"""quenchpath's own science-output modules: pathway identity/ranking
(.path_index) and quench-level ratio computation (.quench_level), both
downstream of the pipeline stages themselves - operating on already-
computed stage output (merged pathway data, saved timescale cubes), not
raw model/network input like quenchpath.chemistry, and reusable outside
the stage that first produces what they consume (see each module's own
docstring).
"""
