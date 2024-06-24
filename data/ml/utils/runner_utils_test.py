from absl.testing import absltest

from data.ml.utils import runner_utils


class RunnerUtilsTest(absltest.TestCase):
    def test_str2bool(self):
        self.assertEqual(runner_utils.str2bool("Yes"), True)
        self.assertEqual(runner_utils.str2bool("True"), True)
        self.assertEqual(runner_utils.str2bool("Y"), True)
        self.assertEqual(runner_utils.str2bool("1"), True)
        self.assertEqual(runner_utils.str2bool("No"), False)
        self.assertEqual(runner_utils.str2bool("False"), False)
        self.assertEqual(runner_utils.str2bool("N"), False)
        self.assertEqual(runner_utils.str2bool("0"), False)

        with self.assertRaises(ValueError):
            runner_utils.str2bool(True)
        with self.assertRaises(ValueError):
            runner_utils.str2bool(0)
        with self.assertRaises(ValueError):
            runner_utils.str2bool(1)
        with self.assertRaises(ValueError):
            runner_utils.str2bool("test")


if __name__ == "__main__":
    absltest.main()
