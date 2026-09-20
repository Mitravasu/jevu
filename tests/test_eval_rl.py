import unittest
from contextlib import redirect_stderr
from io import StringIO

import eval_rl


class EvalRLTests(unittest.TestCase):
    def test_default_sizes_include_supported_maximum(self) -> None:
        self.assertEqual(eval_rl._default_sizes(25, 10), [5, 10, 15, 20, 25])
        self.assertEqual(eval_rl._default_sizes(12, 10), [5, 10, 12])

    def test_accepts_sizes_through_50(self) -> None:
        args = eval_rl.parse_args(
            ["--bundle", "bundle", "--sizes", "30", "40", "50"]
        )

        self.assertEqual(args.sizes, [30, 40, 50])

    def test_rejects_sizes_over_50(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            eval_rl.parse_args(["--bundle", "bundle", "--sizes", "51"])


if __name__ == "__main__":
    unittest.main()
