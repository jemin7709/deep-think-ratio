"""기존 run 산출물 위에서 재현 실험을 수행하는 도구 모음."""

__all__ = ["run_experiment"]


def __getattr__(name: str):
    if name == "run_experiment":
        from src.experiment.think_n import run_experiment

        return run_experiment
    raise AttributeError(name)
