"""Tests for quenchpath.cli.

The CLI is a thin `click` wrapper around already-tested stage functions
(quenchpath.chemistry.*, quenchpath.stages.*) - these tests monkeypatch those
functions rather than doing real I/O, matching this project's convention of
fast, isolated unit tests (see tests/conftest.py's module docstring). What's
checked here is that each subcommand parses its options correctly and calls
the right underlying function with a matching RunConfig/GridResolution.
"""

import click.testing
import pytest

from quenchpath import cli
from quenchpath.config import GridResolution, RunConfig, SpeciesConversion


@pytest.fixture
def runner():
    return click.testing.CliRunner()


def test_read_polynomials_calls_driver(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_and_save_nasa_polynomials",
        lambda raw, processed, gases: calls.append((raw, processed, gases)) or {"H2": {}},
    )

    result = runner.invoke(cli.main, ["read-polynomials"])

    assert result.exit_code == 0, result.output
    assert "Parsed and saved 1 species" in result.output
    assert len(calls) == 1


def test_read_network_calls_driver(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_and_save_venot2019_network",
        lambda raw, processed: calls.append((raw, processed)) or [{}, {}],
    )

    result = runner.invoke(cli.main, ["read-network"])

    assert result.exit_code == 0, result.output
    assert "Parsed and saved 2 reactions" in result.output
    assert len(calls) == 1


def test_convert_builds_run_config_and_calls_driver(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        cli, "convert_mole_fraction_to_number_density", lambda cfg: calls.append(cfg)
    )

    result = runner.invoke(
        cli.main, ["convert", "--planet", "hd189733b", "--exp", "equ", "--mdh", "2", "--tK", "1"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [RunConfig(planet="hd189733b", exp="equ", mdh="2", tK="1")]


def test_convert_defaults_exp_mdh_tk(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        cli, "convert_mole_fraction_to_number_density", lambda cfg: calls.append(cfg)
    )

    result = runner.invoke(cli.main, ["convert", "--planet", "hd189733b"])

    assert result.exit_code == 0, result.output
    assert calls == [RunConfig(planet="hd189733b", exp="kin", mdh="0", tK="1")]


def test_convert_accepts_any_planet_string(monkeypatch, runner):
    # --planet isn't restricted to a hardcoded list (inputs/run_metadata/
    # planets.yaml is a display-name lookup, not a validation whitelist)
    # - any string naming a real raw-run file works.
    calls = []
    monkeypatch.setattr(
        cli, "convert_mole_fraction_to_number_density", lambda cfg: calls.append(cfg)
    )

    result = runner.invoke(cli.main, ["convert", "--planet", "not-a-real-planet"])

    assert result.exit_code == 0, result.output
    assert calls == [RunConfig(planet="not-a-real-planet")]


def test_calc_rates_calls_driver(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(cli, "calc_rates", lambda cfg: calls.append(cfg))

    result = runner.invoke(cli.main, ["calc-rates", "--planet", "hd189733b"])

    assert result.exit_code == 0, result.output
    assert calls == [RunConfig(planet="hd189733b")]


def test_calc_paths_serial_consumes_generator_and_reports_count(monkeypatch, runner):
    calls = []

    def fake_calc_paths_serial(cfg, grid_resolution, conversion):
        calls.append((cfg, grid_resolution, conversion))
        yield from ({} for _ in range(3))

    monkeypatch.setattr(cli, "calc_paths_serial", fake_calc_paths_serial)

    result = runner.invoke(
        cli.main,
        ["calc-paths-serial", "--planet", "hd189733b", "--exp", "equ", "--grid-resolution", "full"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (RunConfig(planet="hd189733b", exp="equ"), GridResolution.FULL, SpeciesConversion.CH4_CO)
    ]
    assert "3 locations" in result.output


def test_calc_paths_serial_conversion_flag_reaches_the_driver(monkeypatch, runner):
    calls = []

    def fake_calc_paths_serial(cfg, grid_resolution, conversion):
        calls.append((cfg, grid_resolution, conversion))
        yield from ()

    monkeypatch.setattr(cli, "calc_paths_serial", fake_calc_paths_serial)

    result = runner.invoke(
        cli.main, ["calc-paths-serial", "--planet", "hd189733b", "--conversion", "co_ch4"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (RunConfig(planet="hd189733b"), GridResolution.SAMPLE, SpeciesConversion.CO_CH4)
    ]
    assert "CO-to-CH4" in result.output


def test_calc_paths_serial_defaults_to_sample(monkeypatch, runner):
    calls = []

    def fake_calc_paths_serial(cfg, grid_resolution, conversion):
        calls.append((cfg, grid_resolution, conversion))
        yield from ()

    monkeypatch.setattr(cli, "calc_paths_serial", fake_calc_paths_serial)

    result = runner.invoke(cli.main, ["calc-paths-serial", "--planet", "hd189733b"])

    assert result.exit_code == 0, result.output
    assert calls == [
        (RunConfig(planet="hd189733b"), GridResolution.SAMPLE, SpeciesConversion.CH4_CO)
    ]


def test_merge_paths_calls_driver_and_reports_summary(monkeypatch, runner):
    calls = []

    class _FakeArray:
        size = 6

    fake_temp_cube = type("C", (), {"data": _FakeArray()})()

    def fake_merge_and_save(cfg, grid_resolution, conversion):
        calls.append((cfg, grid_resolution, conversion))
        return fake_temp_cube, None, None, None, None, {"a": 1, "b": 2, "c": 3}, None

    monkeypatch.setattr(cli, "merge_and_save_paths", fake_merge_and_save)
    monkeypatch.setattr(cli.io, "merged_paths_filename", lambda cfg, gr, conversion: "fake.nc")

    result = runner.invoke(
        cli.main,
        ["merge-paths", "--planet", "hd189733b", "--exp", "kin", "--grid-resolution", "full"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(RunConfig(planet="hd189733b"), GridResolution.FULL, SpeciesConversion.CH4_CO)]
    assert "6 locations" in result.output
    assert "3 pathway indices" in result.output
    assert "fake.nc" in result.output


def test_calc_tau_kin_runs_length_scale_and_ch4_co_only(monkeypatch, runner):
    # --exp kin (the default): tau_adv_ls + both tau_ch4_co_* - not tau_adv_qn,
    # which needs an "equ" cfg (see calc_and_save_adv_qn()'s own
    # docstring).
    calls = []
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_ls",
        lambda cfg: calls.append(("tau_adv_ls", cfg)) or {},
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_qn",
        lambda cfg, species="CH4": calls.append(("tau_adv_qn", cfg)) or {},
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_chem_timescale",
        lambda cfg, gr, chem_kind, conversion: calls.append(
            (f"tau_ch4_co_{chem_kind}", cfg, gr, conversion)
        ),
    )

    result = runner.invoke(cli.main, ["calc-taus", "--planet", "hd189733b"])

    assert result.exit_code == 0, result.output
    cfg = RunConfig(planet="hd189733b")
    assert calls == [
        ("tau_adv_ls", cfg),
        ("tau_ch4_co_rls", cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO),
        ("tau_ch4_co_net", cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO),
    ]
    assert "Computed and saved timescales" in result.output


def test_calc_tau_equ_runs_length_scale_and_quenching_only(monkeypatch, runner):
    # --exp equ: tau_adv_ls + tau_adv_qn - not tau_ch4_co_*, which needs a "kin"
    # cfg (the pathway/rate-network search has no equilibrium-chemistry
    # equivalent).
    calls = []
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_ls",
        lambda cfg: calls.append(("tau_adv_ls", cfg)) or {},
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_qn",
        lambda cfg, species="CH4": calls.append(("tau_adv_qn", cfg)) or {},
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_chem_timescale",
        lambda cfg, gr, chem_kind: calls.append((f"tau_ch4_co_{chem_kind}", cfg, gr)),
    )

    result = runner.invoke(cli.main, ["calc-taus", "--planet", "hd189733b", "--exp", "equ"])

    assert result.exit_code == 0, result.output
    cfg = RunConfig(planet="hd189733b", exp="equ")
    assert calls == [("tau_adv_ls", cfg), ("tau_adv_qn", cfg)]
    assert "Computed and saved timescales" in result.output


class _FakeArray:
    size = 6

    def min(self):
        return 0.0

    def max(self):
        return 123.45


def test_run_pipeline_serial_chains_all_five_stages_with_the_same_config(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        cli,
        "convert_mole_fraction_to_number_density",
        lambda cfg: calls.append(("convert", cfg)),
    )
    monkeypatch.setattr(cli, "calc_rates", lambda cfg: calls.append(("calc_rates", cfg)))

    def fake_calc_paths_serial(cfg, grid_resolution, conversion):
        calls.append(("calc_paths_serial", cfg, grid_resolution, conversion))
        yield from ({} for _ in range(2))

    monkeypatch.setattr(cli, "calc_paths_serial", fake_calc_paths_serial)

    fake_temp_cube = type("C", (), {"data": _FakeArray()})()

    def fake_merge_and_save(cfg, grid_resolution, conversion):
        calls.append(("merge_and_save_paths", cfg, grid_resolution, conversion))
        return fake_temp_cube, None, None, None, None, {"a": 1, "b": 2, "c": 3}, None

    monkeypatch.setattr(cli, "merge_and_save_paths", fake_merge_and_save)

    # --exp kin: the pipeline's own final calc-taus step should run tau_adv_ls
    # + both tau_ch4_co_*, not tau_adv_qn (see _calc_and_save_taus()'s own
    # docstring) - only the 3 valid-for-kin functions are monkeypatched,
    # so calling the missing one would raise instead of silently passing.
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_ls",
        lambda cfg: calls.append(("tau_adv_ls", cfg)) or {},
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_chem_timescale",
        lambda cfg, gr, chem_kind, conversion: calls.append(
            (f"tau_ch4_co_{chem_kind}", cfg, gr, conversion)
        ),
    )

    result = runner.invoke(cli.main, ["run-pipeline-serial", "--planet", "wasp17b", "--exp", "kin"])

    assert result.exit_code == 0, result.output
    cfg = RunConfig(planet="wasp17b", exp="kin")
    assert calls == [
        ("convert", cfg),
        ("calc_rates", cfg),
        ("calc_paths_serial", cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO),
        ("merge_and_save_paths", cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO),
        ("tau_adv_ls", cfg),
        ("tau_ch4_co_rls", cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO),
        ("tau_ch4_co_net", cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO),
    ]
    assert "2 locations" in result.output
    assert "6 locations" in result.output
    assert "3 pathway indices" in result.output
    assert "[5/5] calc-taus: done" in result.output


def test_run_pipeline_serial_exp_equ_skips_straight_to_calc_tau(monkeypatch, runner):
    # No pathway search exists for equilibrium chemistry, and calc-taus's
    # own equ timescales read straight from the raw run - convert/
    # calc-rates/calc-paths-serial/merge-paths must never run for --exp
    # equ. Each is monkeypatched to raise, not just left unpatched, so a
    # regression back to always chaining all 5 steps fails loudly here
    # instead of only wasting compute in a real run.
    def _fail(*args, **kwargs):
        raise AssertionError("must not be called for --exp equ")

    monkeypatch.setattr(cli, "convert_mole_fraction_to_number_density", _fail)
    monkeypatch.setattr(cli, "calc_rates", _fail)
    monkeypatch.setattr(cli, "calc_paths_serial", _fail)
    monkeypatch.setattr(cli, "merge_and_save_paths", _fail)
    monkeypatch.setattr(cli, "calc_and_save_chem_timescale", _fail)

    calls = []
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_ls",
        lambda cfg: calls.append(("tau_adv_ls", cfg)),
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_qn",
        lambda cfg, species="CH4": calls.append(("tau_adv_qn", cfg)),
    )

    result = runner.invoke(cli.main, ["run-pipeline-serial", "--planet", "wasp17b", "--exp", "equ"])

    assert result.exit_code == 0, result.output
    cfg = RunConfig(planet="wasp17b", exp="equ")
    assert calls == [("tau_adv_ls", cfg), ("tau_adv_qn", cfg)]
    assert "needs no pathway search" in result.output


def test_run_pipeline_parallel_chains_all_five_stages_with_the_same_config(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        cli,
        "convert_mole_fraction_to_number_density",
        lambda cfg: calls.append(("convert", cfg)),
    )
    monkeypatch.setattr(cli, "calc_rates", lambda cfg: calls.append(("calc_rates", cfg)))

    def fake_calc_paths_parallel(cfg, grid_resolution, max_workers, conversion):
        calls.append(("calc_paths_parallel", cfg, grid_resolution, max_workers, conversion))
        yield from ({} for _ in range(2))

    monkeypatch.setattr(cli, "calc_paths_parallel", fake_calc_paths_parallel)

    fake_temp_cube = type("C", (), {"data": _FakeArray()})()

    def fake_merge_and_save(cfg, grid_resolution, conversion):
        calls.append(("merge_and_save_paths", cfg, grid_resolution, conversion))
        return fake_temp_cube, None, None, None, None, {"a": 1, "b": 2, "c": 3}, None

    monkeypatch.setattr(cli, "merge_and_save_paths", fake_merge_and_save)

    # --exp kin: see the equivalent comment in
    # test_run_pipeline_serial_chains_all_five_stages_with_the_same_config.
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_ls",
        lambda cfg: calls.append(("tau_adv_ls", cfg)) or {},
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_chem_timescale",
        lambda cfg, gr, chem_kind, conversion: calls.append(
            (f"tau_ch4_co_{chem_kind}", cfg, gr, conversion)
        ),
    )

    result = runner.invoke(
        cli.main,
        [
            "run-pipeline-parallel",
            "--planet",
            "wasp17b",
            "--exp",
            "kin",
            "--grid-resolution",
            "full",
            "--max-workers",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = RunConfig(planet="wasp17b", exp="kin")
    assert calls == [
        ("convert", cfg),
        ("calc_rates", cfg),
        ("calc_paths_parallel", cfg, GridResolution.FULL, 4, SpeciesConversion.CH4_CO),
        ("merge_and_save_paths", cfg, GridResolution.FULL, SpeciesConversion.CH4_CO),
        ("tau_adv_ls", cfg),
        ("tau_ch4_co_rls", cfg, GridResolution.FULL, SpeciesConversion.CH4_CO),
        ("tau_ch4_co_net", cfg, GridResolution.FULL, SpeciesConversion.CH4_CO),
    ]
    assert "2 locations" in result.output
    assert "6 locations" in result.output
    assert "3 pathway indices" in result.output
    assert "[5/5] calc-taus: done" in result.output


def test_run_pipeline_parallel_exp_equ_skips_straight_to_calc_tau(monkeypatch, runner):
    # See the equivalent test on run-pipeline-serial for why.
    def _fail(*args, **kwargs):
        raise AssertionError("must not be called for --exp equ")

    monkeypatch.setattr(cli, "convert_mole_fraction_to_number_density", _fail)
    monkeypatch.setattr(cli, "calc_rates", _fail)
    monkeypatch.setattr(cli, "calc_paths_parallel", _fail)
    monkeypatch.setattr(cli, "merge_and_save_paths", _fail)
    monkeypatch.setattr(cli, "calc_and_save_chem_timescale", _fail)

    calls = []
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_ls",
        lambda cfg: calls.append(("tau_adv_ls", cfg)),
    )
    monkeypatch.setattr(
        cli,
        "calc_and_save_adv_qn",
        lambda cfg, species="CH4": calls.append(("tau_adv_qn", cfg)),
    )

    result = runner.invoke(
        cli.main, ["run-pipeline-parallel", "--planet", "wasp17b", "--exp", "equ"]
    )

    assert result.exit_code == 0, result.output
    cfg = RunConfig(planet="wasp17b", exp="equ")
    assert calls == [("tau_adv_ls", cfg), ("tau_adv_qn", cfg)]
    assert "needs no pathway search" in result.output
