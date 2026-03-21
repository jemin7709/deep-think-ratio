# Model Configs

`models/<name>.yaml` files are thin `lm-eval` config fragments.

Keep only model-specific execution settings here:

- `model`
- `model_args`
- `batch_size`
- `max_batch_size`
- `device`
- `apply_chat_template`
- `fewshot_as_multiturn`

`tasks`, `include_path`, `output_path`, and `log_samples` are injected by the bash runner.
