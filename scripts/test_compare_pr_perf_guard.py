import pytest

from compare_pr_perf_guard import parse_run


@pytest.mark.parametrize(
    ("line", "benchmark", "seconds"),
    [
        ("nTri = 128, time = 8.25e-05 sec", "nTri=128", 0.0000825),
        ("time = 2e-05 sec", "benchmark_1", 0.00002),
        ("nTri = 128, time = 0.125 sec", "nTri=128", 0.125),
    ],
)
def test_parse_timing(tmp_path, line, benchmark, seconds):
    path = tmp_path / "run1.txt"
    path.write_text(line + "\n")
    assert parse_run(path, 1)["benchmarks"] == [
        {"benchmark": benchmark, "time_sec": seconds}
    ]
