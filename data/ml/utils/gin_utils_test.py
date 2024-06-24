import os

from absl.testing import absltest

from data.ml.utils import gin_utils


class GinUtilsTest(absltest.TestCase):
    def test_join_paths(self):
        self.assertEqual(os.path.join("path1", "path2", "path3"), gin_utils.join_paths("path1", ["path2", "path3"]))

    def test_join_paths_list(self):
        expected = [os.path.join("path1", "path2", "path3"), os.path.join("path1", "path2", "path4")]
        self.assertEqual(expected, gin_utils.join_paths("path1", ["path2"], ["path3", "path4"]))

    def test_create_date_paths(self):
        partition = "date=2023-01-24/genre=general"
        expected = [
            "date=2023-01-24/genre=general",
            "date=2023-01-23/genre=general",
            "date=2023-01-22/genre=general",
            "date=2023-01-21/genre=general",
            "date=2023-01-20/genre=general",
        ]
        self.assertEqual(expected, gin_utils.create_date_paths(partition, "2023-01-24", days_to_collect=5))

        partition2 = "date=2023-01-24"
        expected2 = ["date=2023-01-24", "date=2023-01-23", "date=2023-01-22", "date=2023-01-21", "date=2023-01-20"]
        self.assertEqual(expected2, gin_utils.create_date_paths(partition2, "2023-01-24", days_to_collect=5))

    def test_concat_strings(self):
        self.assertEqual("abcdef", gin_utils.concat_strings(["ab", "cd", "ef"]))
        self.assertEqual("ab/cd/ef", gin_utils.concat_strings(["ab", "cd", "ef"], delimiter="/"))


if __name__ == "__main__":
    absltest.main()
