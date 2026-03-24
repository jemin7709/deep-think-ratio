"""현재 프로젝트 산출물 구조에 맞춘 aggregation 도구 모음."""

__all__ = ["aggregate_bins", "make_bins"]


def __getattr__(name: str):
    if name == "aggregate_bins":
        from src.aggregation.aggregate_dtr_pass1_correlation import aggregate_bins

        return aggregate_bins
    if name == "make_bins":
        from src.aggregation.dtr_pass1_correlation import make_bins

        return make_bins
    raise AttributeError(name)
