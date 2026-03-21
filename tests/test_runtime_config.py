import tempfile
import unittest
from pathlib import Path

from scripts.common import (
    build_base_url,
    build_evaluator_config,
    build_run_dir,
    build_tracker_output_path,
    build_vllm_command,
    load_model_settings,
    load_task_settings,
)


class RuntimeConfigTest(unittest.TestCase):
    def test_load_model_settings_requires_new_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat.yaml"
            path.write_text("model: hf\nmodel_args: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new server/harness schema exactly"):
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

    def test_local_chat_completions_forces_batch_size_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "chat.yaml"
            model_path.write_text(
                "\n".join(
                    [
                        "server:",
                        "  model: chat-model",
                        "  host: 127.0.0.1",
                        "  port: 9000",
                        "  chat_template: /tmp/template.jinja",
                        "harness:",
                        "  model: local-chat-completions",
                        "  batch_size: 8",
                        "  apply_chat_template: true",
                        "  predict_only: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            settings = load_model_settings(model_path)
            self.assertEqual(settings.harness.batch_size, 1)
            self.assertEqual(
                build_base_url(settings.server, settings.harness.model),
                "http://127.0.0.1:9000/v1/chat/completions",
            )

    def test_local_chat_completions_requires_chat_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "chat.yaml"
            model_path.write_text(
                "\n".join(
                    [
                        "server:",
                        "  model: chat-model",
                        "harness:",
                        "  model: local-chat-completions",
                        "  batch_size: 1",
                        "  apply_chat_template: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "server.chat_template"):
                load_model_settings(model_path)

    def test_build_vllm_command_uses_server_section(self):
        settings = load_model_settings(Path("models/gpt-oss-120b.yaml"))
        command = build_vllm_command(settings, seed=17)
        self.assertEqual(command[:4], ["vllm", "serve", "openai/gpt-oss-120b", "--host"])
        self.assertIn("--tensor-parallel-size", command)
        self.assertIn("--max-model-len", command)
        self.assertEqual(command[command.index("--seed") + 1], "17")

    def test_build_evaluator_config_uses_results_json_output(self):
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
            self.assertEqual(config.model, "local-completions")
            self.assertEqual(config.model_args["seed"], 11)


if __name__ == "__main__":
    unittest.main()
