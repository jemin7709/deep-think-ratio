import tempfile
import unittest
from pathlib import Path

from src.evaluation.common import (
    TaggedYamlLoader,
    build_evaluator_config,
    build_run_dir,
    build_tracker_output_path,
    load_model_settings,
    load_task_settings,
    load_yaml,
)


class RuntimeConfigTest(unittest.TestCase):
    def test_load_model_settings_rejects_removed_server_harness_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.yaml"
            path.write_text(
                "\n".join(
                    [
                        "server:",
                        "  model: openai/gpt-oss-120b",
                        "harness:",
                        "  model: local-completions",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "removed server/harness schema"):
                load_model_settings(path)

    def test_load_model_settings_requires_vllm_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hf.yaml"
            path.write_text(
                "\n".join(
                    [
                        "model: hf",
                        "model_args:",
                        "  pretrained: openai/gpt-oss-120b",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must declare model: vllm"):
                load_model_settings(path)

    def test_build_run_dir_uses_task_model_seed_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = build_run_dir(
                "aime24_custom",
                "gpt-oss-120b",
                7,
                root=Path(tmpdir),
                timestamp="20260321T000000Z",
            )
            self.assertEqual(
                run_dir,
                Path(tmpdir) / "aime24_custom" / "gpt-oss-120b" / "7" / "20260321T000000Z",
            )
            self.assertEqual(build_tracker_output_path(run_dir), run_dir / "results.json")

    def test_build_evaluator_config_uses_vllm_backend_and_injects_seed(self):
        task_settings = load_task_settings(Path("tasks/aime24/aime24_custom.yaml"))
        model_settings = load_model_settings(Path("models/gpt-oss-120b.yaml"))
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = build_evaluator_config(
                task_settings,
                model_settings,
                run_dir=run_dir,
                seed=11,
                limit=0.25,
            )

        self.assertEqual(config.output_path, str(run_dir / "results.json"))
        self.assertEqual(config.tasks, ["aime24_custom"])
        self.assertEqual(config.model, "vllm")
        self.assertEqual(config.model_args["pretrained"], "openai/gpt-oss-120b")
        self.assertEqual(config.model_args["seed"], 11)
        self.assertEqual(
            config.model_args["chat_template_args"]["reasoning_effort"],
            "medium",
        )
        self.assertEqual(config.batch_size, "auto")
        self.assertEqual(config.metadata["seed"], 11)
        self.assertEqual(config.metadata["model"], "vllm")
        self.assertEqual(config.metadata["model_args"]["seed"], 11)
        self.assertNotIn("server_seed", config.metadata)

    def test_aime24_prompt_keeps_literal_boxed_marker(self):
        raw = load_yaml(
            Path("tasks/aime24/aime24_custom.yaml"),
            loader=TaggedYamlLoader,
        )

        self.assertIn(r"\boxed{}", raw["doc_to_text"])
        self.assertNotIn("\b", raw["doc_to_text"])

    def test_load_model_settings_leaves_batch_size_unset_when_omitted(self):
        settings = load_model_settings(Path("models/dummy.yaml"))
        self.assertIsNone(settings.batch_size)


if __name__ == "__main__":
    unittest.main()
