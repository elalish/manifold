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
