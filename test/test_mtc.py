# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import workflow


def _example_path(dirname):
    """Paths to things in the top-level directory of this repository."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", dirname))


def _test_path(dirname):
    """Paths to things in the `test` directory."""
    return os.path.join(os.path.dirname(__file__), dirname)


def run_test_mtc(
    multiprocess=False, chunkless=False, recode=False, sharrow=False, extended=False
):
    def regress(ext, out_dir):
        if ext:
            regress_trips_df = pd.read_csv(_test_path("regress/final_trips-ext.csv"))
        else:
            regress_trips_df = pd.read_csv(_test_path("regress/final_trips.csv"))
        final_trips_df = pd.read_csv(_test_path(out_dir.joinpath("final_trips.csv")))

        # column order may not match, so fix it before checking
        assert sorted(regress_trips_df.columns) == sorted(final_trips_df.columns)
        final_trips_df = final_trips_df[regress_trips_df.columns]

        # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
        # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
        # compare_cols = []
        pdt.assert_frame_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    run_args = []

    if multiprocess:
        if extended:
            run_args.extend(
                [
                    "-c",
                    _test_path("ext-configs_mp"),
                    "-c",
                    _example_path("ext-configs_mp"),
                ]
            )
        else:
            run_args.extend(
                [
                    "-c",
                    _test_path("configs_mp"),
                    "-c",
                    _example_path("configs_mp"),
                ]
            )
    elif chunkless:
        if extended:
            run_args.extend(
                [
                    "-c",
                    _test_path("ext-configs_chunkless"),
                ]
            )
        else:
            run_args.extend(
                [
                    "-c",
                    _test_path("configs_chunkless"),
                ]
            )
    elif recode:
        run_args.extend(
            [
                "-c",
                _test_path("configs_recode"),
            ]
        )
    elif sharrow:
        run_args.extend(
            [
                "-c",
                _test_path("configs_sharrow"),
            ]
        )
        if os.environ.get("GITHUB_ACTIONS") != "true":
            run_args.append("--persist-sharrow-cache")
    else:
        run_args.extend(
            [
                "-c",
                _test_path("configs"),
            ]
        )

    # general run args
    if extended:
        run_args.extend(
            [
                "--data_model",
                _example_path("data_model"),
                "-c",
                _test_path("ext-configs"),
                "-c",
                _example_path("ext-configs"),
            ]
        )

    out_dir = _test_path(
        f"output-{'mp' if multiprocess else 'single'}"
        f"-{'chunkless' if chunkless else 'chunked'}"
        f"-{'recode' if recode else 'no_recode'}"
        f"-{'sharrow' if sharrow else 'no_sharrow'}"
        f"-{'ext' if extended else 'no_ext'}"
    )

    # create output directory if it doesn't exist and add .gitignore
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir).joinpath(".gitignore").write_text("**\n")

    run_args.extend(
        [
            "-c",
            _example_path("configs"),
            "-d",
            _example_path("data"),
            "-o",
            out_dir,
        ]
    )

    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run([sys.executable, file_path] + run_args, check=True)

    regress(extended, Path(out_dir))


EXPECTED_MODELS = [
    "input_checker",
    "initialize_proto_population",
    "compute_disaggregate_accessibility",
    "initialize_landuse",
    "initialize_households",
    "compute_accessibility",
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "vehicle_type_choice",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "school_escorting",
    "joint_tour_frequency",
    "joint_tour_composition",
    "joint_tour_participation",
    "joint_tour_destination",
    "joint_tour_scheduling",
    "non_mandatory_tour_frequency",
    "non_mandatory_tour_destination",
    "non_mandatory_tour_scheduling",
    "vehicle_allocation",
    "tour_mode_choice_simulate",
    "atwork_subtour_frequency",
    "atwork_subtour_destination",
    "atwork_subtour_scheduling",
    "atwork_subtour_mode_choice",
    "stop_frequency",
    "trip_purpose",
    "trip_destination",
    "trip_purpose_and_destination",
    "trip_scheduling",
    "trip_mode_choice",
    "write_data_dictionary",
    "track_skim_usage",
    "write_trip_matrices",
    "write_tables",
]


@pytest.mark.parametrize(
    "chunk_training_mode,recode_pipeline_columns,sharrow_enabled",
    [
        ("disabled", True, False),
        ("explicit", False, False),
        ("explicit", True, True),
    ],
)
def test_mtc_extended_progressive(
    chunk_training_mode, recode_pipeline_columns, sharrow_enabled
):
    import activitysim.abm  # register components # noqa: F401

    out_dir = _test_path(f"output-progressive-recode{recode_pipeline_columns}")
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir).joinpath(".gitignore").write_text("**\n")

    working_dir = Path(_example_path("."))

    output_trips_table = {"tablename": "trips"}
    if recode_pipeline_columns:
        output_trips_table["decode_columns"] = {
            "origin": "land_use.zone_id",
            "destination": "land_use.zone_id",
        }

    settings = {
        "treat_warnings_as_errors": False,
        "households_sample_size": 10,
        "chunk_size": 0,
        "chunk_training_mode": chunk_training_mode,
        "use_shadow_pricing": False,
        "want_dest_choice_sample_tables": False,
        "cleanup_pipeline_after_run": True,
        "output_tables": {
            "h5_store": False,
            "action": "include",
            "prefix": "final_",
            "sort": True,
            "tables": [
                output_trips_table,
            ],
        },
        "recode_pipeline_columns": recode_pipeline_columns,
        "trace_hh_id": 1196298,
    }

    if sharrow_enabled and not recode_pipeline_columns:
        raise ValueError("sharrow_enabled requires recode_pipeline_columns")

    if sharrow_enabled:
        settings["sharrow"] = "test"  # check sharrow in `test` mode
        del settings["trace_hh_id"]  # do not test sharrow with tracing

    state = workflow.State.make_default(
        working_dir=working_dir,
        configs_dir=("ext-configs", "configs"),
        data_dir="data",
        data_model_dir="data_model",
        output_dir=out_dir,
        settings=settings,
    )
    state.filesystem.persist_sharrow_cache()
    state.logging.config_logger()

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.chunk_size == 0
    if not sharrow_enabled:
        assert not state.settings.sharrow

    ref_pipeline = Path(__file__).parent.joinpath(
        f"reference-pipeline-extended-recode{recode_pipeline_columns}.zip"
    )
    if not ref_pipeline.exists():
        # if reference pipeline does not exist, don't clean up so we can save
        # and create it at the end of running this test.
        state.settings.cleanup_pipeline_after_run = False

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        if ref_pipeline.exists():
            try:
                state.checkpoint.check_against(
                    Path(__file__).parent.joinpath(
                        f"reference-pipeline-extended-recode{recode_pipeline_columns}.zip"
                    ),
                    checkpoint_name=step_name,
                )
            except Exception:
                print(f"> prototype_mtc_extended {step_name}: ERROR")
                raise
            else:
                print(f"> prototype_mtc_extended {step_name}: ok")
        else:
            print(f"> prototype_mtc_extended {step_name}: ran, not checked")

    if not ref_pipeline.exists():
        # make new reference pipeline file if it is missing
        import shutil

        shutil.make_archive(
            ref_pipeline.with_suffix(""), "zip", state.checkpoint.store.filename
        )


if __name__ == "__main__":
    test_mtc_extended_progressive("disabled", True, False)
    test_mtc_extended_progressive("explicit", False, False)
    test_mtc_extended_progressive("explicit", True, True)
