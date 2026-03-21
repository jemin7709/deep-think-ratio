# Model Configs

`models/<name>.yaml` files now map directly to `lm-eval`'s `vllm` backend.

The old `server/harness` schema is removed. API-server based configs are rejected.

Example:

```yaml
model: vllm
model_args:
  pretrained: openai/gpt-oss-120b
  dtype: auto
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 131072
  chat_template_args:
    reasoning_effort: medium
batch_size: auto
apply_chat_template: true
fewshot_as_multiturn: true
predict_only: true
gen_kwargs:
  temperature: 1.0
  top_p: 1.0
  max_gen_toks: 131072
```

Notes:

- `model` must be `vllm`.
- `model_args` are forwarded to `lm-eval`'s native `vllm` backend.
- `model_args.chat_template_args.reasoning_effort` is the supported place to control GPT-OSS reasoning effort.
- `batch_size` is optional. Omit it to defer to the installed `lm-eval` default, or set it to `auto` explicitly.
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
