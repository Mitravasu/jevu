import unittest
from unittest.mock import patch

from jevu.rl.device import resolve_device


class RLDeviceTests(unittest.TestCase):
    @patch("jevu.rl.device._device_available")
    def test_auto_prefers_apple_mps(self, available) -> None:
        available.side_effect = lambda device: device in {"mps", "cpu"}

        self.assertEqual(resolve_device("auto"), "mps")
        available.assert_called_once_with("mps")

    @patch("jevu.rl.device._device_available")
    def test_auto_falls_back_to_cuda_then_cpu(self, available) -> None:
        available.side_effect = lambda device: device in {"cuda", "cpu"}
        self.assertEqual(resolve_device("auto"), "cuda")

        available.reset_mock()
        available.side_effect = lambda device: device == "cpu"
        self.assertEqual(resolve_device("auto"), "cpu")

    @patch("jevu.rl.device._device_available", return_value=False)
    def test_explicit_unavailable_device_fails(self, _available) -> None:
        with self.assertRaisesRegex(RuntimeError, "unavailable: mps"):
            resolve_device("mps")


if __name__ == "__main__":
    unittest.main()
