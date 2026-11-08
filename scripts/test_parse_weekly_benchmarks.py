import json

from parse_weekly_benchmarks import parse_gtest_suite


def write_run(path, sometimes_result, sometimes_time):
    records = [
        ("Sometimes", "RUN", sometimes_result, sometimes_time),
        ("DISABLED_Never", "NOTRUN", "SUPPRESSED", "0s"),
        ("Fast", "RUN", "COMPLETED", "0s"),
    ]
    tests = [
        dict(classname="Suite", name=name, status=status, result=result, time=time)
        for name, status, result, time in records
    ]
    path.write_text(json.dumps({"testsuites": [{"testsuite": tests}]}))


def test_mixed_skips_count_only_completed_runs(tmp_path):
    write_run(tmp_path / "run1.json", "SKIPPED", "0s")
    write_run(tmp_path / "run2.json", "COMPLETED", "0.024s")
    suite = parse_gtest_suite(tmp_path)
    timings = {test["name"]: test["time_ms"] for test in suite["tests"]}
    assert timings["Suite.Sometimes"]["samples_ms"] == [24.0]
    assert timings["Suite.Fast"]["samples_ms"] == [0.0, 0.0]
    assert "Suite.DISABLED_Never" not in suite["test_order"]
