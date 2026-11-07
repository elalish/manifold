import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import parse_weekly_benchmarks as parser


class EmberCaseAlignmentTests(unittest.TestCase):
    def parse(self, case_runs):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, cases in enumerate(case_runs, 1):
                (root / f"run{i}.txt").write_text("".join(
                    f"### case {index}\n--- {ms} ms for Winding03 P\n" for index, ms in cases
                ))
            specs = [{"id_a": "a", "id_b": "b"}] * 2
            with patch.object(parser, "load_ember_specs", return_value=specs):
                return parser.parse_ember_suite(root, root)

    def test_reordered_cases(self):
        report = self.parse([[(0, 10), (1, 20)], [(1, 21), (0, 11)]])
        self.assertEqual(report["cases"][0]["phases"]["Winding03 P"]["samples_ms"], [10, 11])

    def test_inconsistent_case_sets(self):
        for cases in [[(0, 1)], [(0, 1), (0, 2)], [(0, 1), (1, 2), (2, 3)]]:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.parse([[(0, 10), (1, 20)], cases])
        with self.assertRaisesRegex(ValueError, "Unknown Ember case index"):
            self.parse([[(2, 1)]])



class GoogleTestAlignmentTests(unittest.TestCase):
    def test_reordered_and_skipped_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, names in enumerate([["A", "B"], ["B", "A"]], 1):
                results = [
                    {"classname": "Suite", "name": name, "time": "1s" if name == "A" else "2s"}
                    for name in names
                ]
                results.extend([
                    {"classname": "Suite", "name": "Disabled", "status": "NOTRUN"},
                    {"classname": "Suite", "name": "Skipped", "result": "SKIPPED"},
                ])
                (root / f"run{i}.json").write_text(json.dumps({"testsuites": [{"testsuite": results}]}))
            report = parser.parse_gtest_suite(root)
            self.assertEqual(report["test_order"], ["Suite.A", "Suite.B"])
            self.assertEqual(report["tests"][0]["time_ms"]["samples_ms"], [1000.0, 1000.0])
