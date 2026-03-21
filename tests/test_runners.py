import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lm_eval.loggers import EvaluationTracker

from scripts.eval import save_raw_results, validate_repeats
from scripts.run_many import run_one


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
    @patch("scripts.run_many.stop_server")
    @patch("scripts.run_many.write_postprocess_artifacts")
    @patch("scripts.run_many.run_evaluation")
    @patch("scripts.run_many.wait_for_server")
    @patch("scripts.run_many.start_server")
    @patch("scripts.run_many.build_run_dir")
    def test_run_one_orders_steps(
        self,
        build_run_dir_mock,
        start_server_mock,
        wait_for_server_mock,
        run_evaluation_mock,
        write_postprocess_mock,
        stop_server_mock,
    ):
        process = Mock()
        run_dir = Path("/tmp/results/aime24_custom/model/7/stamp")
        build_run_dir_mock.return_value = run_dir
        start_server_mock.return_value = (process, "http://127.0.0.1:8000/health")

        returned_run_dir = run_one(
            task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
            model_config_path=Path("models/dummy.yaml"),
            seed=7,
            limit=0.1,
        )

        self.assertEqual(returned_run_dir, run_dir)
        wait_for_server_mock.assert_called_once_with(
            process, "http://127.0.0.1:8000/health"
        )
        run_evaluation_mock.assert_called_once_with(
            task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
            model_config_path=Path("models/dummy.yaml"),
            seed=7,
            run_dir=run_dir,
            limit=0.1,
        )
        write_postprocess_mock.assert_called_once_with(run_dir=run_dir)
        stop_server_mock.assert_called_once_with(process)

    @patch("scripts.run_many.stop_server")
    @patch("scripts.run_many.run_evaluation", side_effect=RuntimeError("boom"))
    @patch("scripts.run_many.wait_for_server")
    @patch("scripts.run_many.start_server")
    @patch("scripts.run_many.build_run_dir")
    def test_run_one_cleans_up_on_eval_failure(
        self,
        build_run_dir_mock,
        start_server_mock,
        wait_for_server_mock,
        run_evaluation_mock,
        stop_server_mock,
    ):
        process = Mock()
        build_run_dir_mock.return_value = Path("/tmp/results/aime24_custom/model/7/stamp")
        start_server_mock.return_value = (process, "http://127.0.0.1:8000/health")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_one(
                task_config_path=Path("tasks/aime24/aime24_custom.yaml"),
                model_config_path=Path("models/dummy.yaml"),
                seed=7,
                limit=None,
            )

        stop_server_mock.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
