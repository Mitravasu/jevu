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

    def test_resume_source_can_be_selected(self) -> None:
        args = train_rl.parse_args(
            ["--config", "training.toml", "--resume", "artifacts/rl/run"]
        )

        self.assertEqual(args.resume, "artifacts/rl/run")

    def test_warm_start_source_can_be_selected(self) -> None:
        args = train_rl.parse_args(
            ["--config", "training.toml", "--warm-start", "artifacts/rl/run"]
        )

        self.assertEqual(args.warm_start, "artifacts/rl/run")

    def test_resume_and_warm_start_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            train_rl.parse_args(
                [
                    "--config",
                    "training.toml",
                    "--resume",
                    "resume-run",
                    "--warm-start",
                    "warm-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
