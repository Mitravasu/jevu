import unittest

import train_rl


class TrainRLTests(unittest.TestCase):
    def test_console_output_can_be_overridden(self) -> None:
        args = train_rl.parse_args(
            [
                "--config",
                "training.toml",
                "--console-output",
                "quiet",
            ]
        )

        self.assertEqual(args.console_output, "quiet")


if __name__ == "__main__":
    unittest.main()
