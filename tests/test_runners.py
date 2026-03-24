import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from lm_eval.loggers import EvaluationTracker

from run import run_one
from src.evaluation.eval import save_raw_results, validate_repeats


class EvalRunnerTest(unittest.TestCase):
    def test_save_raw_results_calls_manual_tracker_saves(self):
        tracker = Mock(spec=EvaluationTracker)
        results = {
            "config": {"model": "dummy"},
            "samples": {
                "aime24_custom": [
                    {"doc_id": 0, "resps": [["42", "0"]], "filtered_resps": [["42"]]}
                ]
            },
        }

        returned_samples = save_raw_results(results, tracker)

        tracker.save_results_aggregated.assert_called_once_with(
            results={"config": {"model": "dummy"}},
            samples=returned_samples,
        )
        tracker.save_results_samples.assert_called_once_with(
            task_name="aime24_custom",
            samples=returned_samples["aime24_custom"],
        )

    def test_validate_repeats_rejects_mismatch(self):
        with self.assertRaisesRegex(ValueError, "expected 4 completions"):
            validate_repeats(
                {"aime24_custom": [{"doc_id": 0, "resps": [["42", "0"]]}]},
                task_name="aime24_custom",
                expected_repeats=4,
            )


class RunManyTest(unittest.TestCase):
    @patch("run.write_postprocess_artifacts")
    @patch("run.run_evaluation")
    @patch("run.build_run_dir")
    def test_run_one_orders_steps(
        self,
        build_run_dir_mock,
        run_evaluation_mock,
        write_postprocess_mock,
    ):
        run_dir = Path("/tmp/results/aime24_custom/model/7/stamp")
        build_run_dir_mock.return_value = run_dir

        returned_run_dir = run_one(
            task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
            model_config_path=Path("models/dummy.yaml"),
            seed=7,
            limit=0.1,
        )

        self.assertEqual(returned_run_dir, run_dir)
        run_evaluation_mock.assert_called_once_with(
            task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
            model_config_path=Path("models/dummy.yaml"),
            seed=7,
            run_dir=run_dir,
            limit=0.1,
        )
        write_postprocess_mock.assert_called_once_with(run_dir=run_dir)

    @patch("run.write_postprocess_artifacts")
    @patch("run.run_evaluation", side_effect=RuntimeError("boom"))
    @patch("run.build_run_dir")
    def test_run_one_does_not_postprocess_on_eval_failure(
        self,
        build_run_dir_mock,
        run_evaluation_mock,
        write_postprocess_mock,
    ):
        build_run_dir_mock.return_value = Path(
            "/tmp/results/aime24_custom/model/7/stamp"
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_one(
                task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
                model_config_path=Path("models/dummy.yaml"),
                seed=7,
                limit=None,
            )

        run_evaluation_mock.assert_called_once()
        write_postprocess_mock.assert_not_called()

    @patch("run.write_postprocess_artifacts")
    @patch("run.run_evaluation", side_effect=RuntimeError("boom"))
    @patch("run.build_run_dir")
    def test_run_one_deletes_empty_run_dir_on_failure(
        self,
        build_run_dir_mock,
        run_evaluation_mock,
        write_postprocess_mock,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir) / "results" / "aime24_custom" / "model" / "7" / "stamp"
            )
            run_dir.mkdir(parents=True)
            build_run_dir_mock.return_value = run_dir

            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_one(
                    task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
                    model_config_path=Path("models/dummy.yaml"),
                    seed=7,
                    limit=None,
                )

            self.assertFalse(run_dir.exists())
            run_evaluation_mock.assert_called_once()
            write_postprocess_mock.assert_not_called()

    @patch("run.write_postprocess_artifacts")
    @patch("run.build_run_dir")
    def test_run_one_keeps_partial_artifacts_on_failure(
        self,
        build_run_dir_mock,
        write_postprocess_mock,
    ):
        def write_partial_results(**kwargs):
            run_dir = kwargs["run_dir"]
            (run_dir / "results_2026-03-22T00-00-00.json").write_text(
                "{}\n", encoding="utf-8"
            )
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir) / "results" / "aime24_custom" / "model" / "7" / "stamp"
            )
            run_dir.mkdir(parents=True)
            build_run_dir_mock.return_value = run_dir

            with patch("run.run_evaluation", side_effect=write_partial_results):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    run_one(
                        task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
                        model_config_path=Path("models/dummy.yaml"),
                        seed=7,
                        limit=None,
                    )

            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "results_2026-03-22T00-00-00.json").is_file())
            write_postprocess_mock.assert_not_called()

    @patch("run.write_postprocess_artifacts")
    @patch("run.run_evaluation", side_effect=KeyboardInterrupt())
    @patch("run.build_run_dir")
    def test_run_one_deletes_empty_run_dir_on_keyboard_interrupt(
        self,
        build_run_dir_mock,
        run_evaluation_mock,
        write_postprocess_mock,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = (
                Path(tmpdir) / "results" / "aime24_custom" / "model" / "7" / "stamp"
            )
            run_dir.mkdir(parents=True)
            build_run_dir_mock.return_value = run_dir

            with self.assertRaises(KeyboardInterrupt):
                run_one(
                    task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
                    model_config_path=Path("models/dummy.yaml"),
                    seed=7,
                    limit=None,
                )

            self.assertFalse(run_dir.exists())
            run_evaluation_mock.assert_called_once()
            write_postprocess_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
