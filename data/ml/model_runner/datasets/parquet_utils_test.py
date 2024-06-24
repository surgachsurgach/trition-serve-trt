import os
import tempfile

from absl.testing import absltest
from pyarrow import parquet

from data.ml.model_runner.datasets import parquet_utils
from data.ml.utils import file_utils


class ParquetUtilsTest(absltest.TestCase):
    def test_make_parquet_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_file_name = os.path.join(tmpdirname, "test.snappy.parquet")
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
                parquet_file_name,
            )

            actual = parquet.read_table(parquet_file_name).to_pydict()
            expected = {"col1": [1, 2, 3], "col2": [5, 6, 7]}

            self.assertEqual(expected, actual)

    def test_get_parquet_files(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_file_name1 = os.path.join(tmpdirname, "test1.snappy.parquet")
            parquet_file_name2 = os.path.join(tmpdirname, "test2.snappy.parquet")
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
                parquet_file_name1,
            )
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
                parquet_file_name2,
            )

            filesystem = file_utils.get_filesystem(tmpdirname)

            files = parquet_utils.get_parquet_files(tmpdirname, shuffle=False)
            self.assertEqual([filesystem.unstrip_protocol(parquet_file_name1), filesystem.unstrip_protocol(parquet_file_name2)], files)

            files_shuffled = parquet_utils.get_parquet_files(tmpdirname, shuffle=True, seed=1234)
            self.assertEqual(
                [filesystem.unstrip_protocol(parquet_file_name2), filesystem.unstrip_protocol(parquet_file_name1)], files_shuffled
            )

    def test_get_num_rows(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_file_name_1 = os.path.join(tmpdirname, "test1.snappy.parquet")
            parquet_file_name_2 = os.path.join(tmpdirname, "test2.snappy.parquet")
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
                parquet_file_name_1,
            )
            parquet_utils.make_parquet_file(
                {"col1": [10, 11], "col2": [15, 16]},
                parquet_file_name_2,
            )

            self.assertEqual(5, parquet_utils.get_num_rows([parquet_file_name_1, parquet_file_name_2]))

    def test_get_filtered_num_rows(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_file_name_1 = os.path.join(tmpdirname, "test1.snappy.parquet")
            parquet_file_name_2 = os.path.join(tmpdirname, "test2.snappy.parquet")
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
                parquet_file_name_1,
            )
            parquet_utils.make_parquet_file(
                {"col1": [10, 11], "col2": [15, 16]},
                parquet_file_name_2,
            )

            self.assertEqual(4, parquet_utils.get_filtered_num_rows([parquet_file_name_1, parquet_file_name_2], "col2", lambda x: x > 5))
            self.assertEqual(1, parquet_utils.get_filtered_num_rows([parquet_file_name_1, parquet_file_name_2], "col2", lambda x: x <= 5))

    def test_iter_rows(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            parquet_file_name_1 = os.path.join(tmpdirname, "test1.snappy.parquet")
            parquet_file_name_2 = os.path.join(tmpdirname, "test2.snappy.parquet")
            parquet_utils.make_parquet_file(
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
                parquet_file_name_1,
            )
            parquet_utils.make_parquet_file(
                {"col1": [10, 11], "col2": [15, 16]},
                parquet_file_name_2,
            )

            actual = list(parquet_utils.iter_rows([parquet_file_name_1, parquet_file_name_2]))
            expected = [
                {"col1": 1, "col2": 5},
                {"col1": 2, "col2": 6},
                {"col1": 3, "col2": 7},
                {"col1": 10, "col2": 15},
                {"col1": 11, "col2": 16},
            ]

            self.assertEqual(expected, actual)


if __name__ == "__main__":
    absltest.main()
