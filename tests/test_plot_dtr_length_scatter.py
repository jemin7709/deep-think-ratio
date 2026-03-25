import json
import tempfile
import unittest
from pathlib import Path

from src.aggregation.dtr_length_scatter import (
    DEFAULT_OUTPUT_DIR_NAME,
    OUTPUT_JSON_FILENAME,
    OUTPUT_PLOT_FILENAME,
    build_title,
    default_output_dir,
    load_points,
    pearson_r,
    resolve_model_name,
    resolve_paths,
    write_summary_json,
)
from src.dtr.jsd_utils import dtr_results_path


class PlotDtrLengthScatterTest(unittest.TestCase):
    def test_load_points_reads_dtr_and_num_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtr_path = Path(tmpdir) / "dtr.json"
            dtr_path.write_text(
                json.dumps(
                    [
                        {"doc_id": 3, "repeat_index": 1, "dtr": 0.7, "num_tokens": 12},
                        {"doc_id": 2, "repeat_index": 0, "dtr": 0.2, "num_tokens": 5},
                    ]
                ),
                encoding="utf-8",
            )

            points = load_points(dtr_path)

            self.assertEqual(
                [
                    (point.doc_id, point.repeat_index, point.dtr, point.response_length)
                    for point in points
                ],
                [(2, 0, 0.2, 5), (3, 1, 0.7, 12)],
            )

    def test_load_points_requires_num_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtr_path = Path(tmpdir) / "dtr.json"
            dtr_path.write_text(
                json.dumps([{"doc_id": 0, "repeat_index": 0, "dtr": 0.1}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing num_tokens"):
                load_points(dtr_path)

    def test_write_summary_json_matches_expected_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            dtr_path = dtr_results_path(run_dir)
            output_path = default_output_dir(run_dir) / OUTPUT_JSON_FILENAME
            results_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "model_args": {"pretrained": "openai/gpt-oss-120b"},
                        },
                        "results": {"aime24_custom": {}},
                    }
                ),
                encoding="utf-8",
            )
            dtr_path.parent.mkdir(parents=True, exist_ok=True)
            dtr_path.write_text(
                json.dumps(
                    [
                        {"doc_id": 0, "repeat_index": 0, "dtr": 0.1, "num_tokens": 10},
                        {"doc_id": 0, "repeat_index": 1, "dtr": 0.3, "num_tokens": 20},
                        {"doc_id": 1, "repeat_index": 0, "dtr": 0.6, "num_tokens": 40},
                    ]
                ),
                encoding="utf-8",
            )

            points = load_points(dtr_path)
            pearson = pearson_r(
                [point.dtr for point in points],
                [float(point.response_length) for point in points],
            )
            write_summary_json(
                run_dir=run_dir,
                task_name="aime24_custom",
                model_name="openai/gpt-oss-120b",
                dtr_path=dtr_path,
                results_path=results_path,
                output_path=output_path,
                points=points,
                pearson=pearson,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(
                default_output_dir(run_dir), run_dir / DEFAULT_OUTPUT_DIR_NAME
            )
            self.assertEqual(summary["task"], "aime24_custom")
            self.assertEqual(summary["model"], "openai/gpt-oss-120b")
            self.assertEqual(summary["num_sequences"], 3)
            self.assertAlmostEqual(summary["pearson_r"], pearson)
            self.assertEqual(summary["dtr_min"], 0.1)
            self.assertEqual(summary["dtr_max"], 0.6)
            self.assertEqual(summary["length_min"], 10)
            self.assertEqual(summary["length_max"], 40)
            self.assertAlmostEqual(summary["length_mean"], 70 / 3)
            self.assertEqual(len(summary["points"]), 3)

    def test_resolve_paths_uses_default_output_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            results_path = run_dir / "results_2026-03-22T00-00-00.json"
            results_path.write_text(
                json.dumps({"results": {"aime24_custom": {}}}),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "run_dir": run_dir,
                    "dtr_path": None,
                    "results_path": None,
                    "output_plot": None,
                    "output_json": None,
                },
            )()

            dtr_path, _results_path, output_plot, output_json = resolve_paths(args)

            self.assertEqual(dtr_path, dtr_results_path(run_dir))
            self.assertEqual(
                output_plot, default_output_dir(run_dir) / OUTPUT_PLOT_FILENAME
            )
            self.assertEqual(
                output_json, default_output_dir(run_dir) / OUTPUT_JSON_FILENAME
            )

    def test_resolve_model_name_prefers_pretrained_config(self):
        run_dir = Path("/tmp/results/aime24_custom/model/0/2026-03-25T00-00-00")
        aggregated = {"config": {"model_args": {"pretrained": "openai/gpt-oss-120b"}}}

        self.assertEqual(resolve_model_name(aggregated, run_dir), "openai/gpt-oss-120b")
        self.assertEqual(
            build_title(run_dir, "aime24_custom", "model", None),
            f"{run_dir.name} | aime24_custom | model | DTR vs Response Length",
        )


if __name__ == "__main__":
    unittest.main()
