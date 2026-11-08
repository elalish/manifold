import pytest

from compare_pr_perf_guard import parse_run, parse_suite


TIMING = "nTri = 128, time = 8.25e-05 sec\n"
RSS = "PEAK_RSS nTri=128 size_index=0 peak_rss_mb=2.00 peak_rss_bytes=2097152\n"


def test_exponent_timing_matches_rss(tmp_path):
    path = tmp_path / "run1.txt"
    path.write_text(TIMING + RSS)
    assert parse_run(path, 1)["benchmarks"] == [
        {"benchmark": "nTri=128", "time_sec": 0.0000825, "peak_rss_mb": 2.0}
    ]


def test_duplicate_rss_identifies_benchmark_and_file(tmp_path):
    path = tmp_path / "run1.txt"
    path.write_text("nTri = 128, time = 0.012 sec\n" + RSS + RSS)
    with pytest.raises(RuntimeError, match="Duplicate peak RSS for nTri=128") as exc:
        parse_run(path, 1)
    assert str(path) in str(exc.value)


def test_partial_rss_preserves_timing_samples(tmp_path):
    timing = "nTri = 128, time = 0.012 sec\n"
    (tmp_path / "run1.txt").write_text(timing + RSS)
    (tmp_path / "run2.txt").write_text(timing)
    result = parse_suite(tmp_path)["benchmarks"]["nTri=128"]
    assert result["timing_sec"]["samples"] == [0.012, 0.012]
    assert result["peak_rss_mb"]["samples"] == [2.0]


def test_missing_all_rss_identifies_benchmark(tmp_path):
    (tmp_path / "run1.txt").write_text("nTri = 128, time = 0.012 sec\n")
    with pytest.raises(RuntimeError, match="peak RSS samples for benchmark 'nTri=128'"):
        parse_suite(tmp_path)
