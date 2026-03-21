# Model Configs

`models/<name>.yaml` files define two distinct layers:

- `server`: how to start `vllm serve`
- `harness`: how `lm-eval` should call the OpenAI-compatible endpoint

The new schema is mandatory. Flat legacy configs are rejected.

Example:

```yaml
server:
  model: openai/gpt-oss-120b
  host: 127.0.0.1
  port: 8000
  dtype: auto
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 131072

harness:
  model: local-completions
  model_args: {}
  batch_size: 1
  apply_chat_template: true
  fewshot_as_multiturn: true
  predict_only: true
  gen_kwargs:
    temperature: 1.0
    top_p: 1.0
    max_gen_toks: 131072
```

Notes:

- `harness.model` defaults to `local-completions`.
- `local-chat-completions` is opt-in only.
- `local-chat-completions` requires:
  - `apply_chat_template: true`
  - `batch_size: 1`
  - `server.chat_template`
- `task.repeats` is the only public repeat setting. Do not set `gen_kwargs.n`.

Run the full workflow with:

```bash
uv run python scripts/run_many.py \
  --task-config tasks/aime24/aime24_custom.yaml \
  --model-config models/gpt-oss-120b.yaml \
  --seed 7
```

For a single run directory without orchestration:

```bash
uv run python scripts/eval.py \
  --task-config tasks/aime24/aime24_custom.yaml \
  --model-config models/gpt-oss-120b.yaml \
  --seed 7 \
  --run-dir results/aime24_custom/gpt-oss-120b/7/manual
```
