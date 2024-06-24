from absl.testing import absltest

from data.ml.utils import file_utils


class FileUtilsTest(absltest.TestCase):
    def test_get_path(self):
        assert "/is/bucket" == file_utils.get_path("s3://bucket/is/bucket")
        assert "/this/is/local" == file_utils.get_path("file:///this/is/local")
        assert "/this/is/local" == file_utils.get_path("/this/is/local")

    def test_get_scheme(self):
        assert "s3" == file_utils.get_scheme("s3://this/is/bucket")
        assert "file" == file_utils.get_scheme("/absolute/path")
        assert "file" == file_utils.get_scheme("relative/path")

        with self.assertRaises(ValueError):
            file_utils.get_scheme("gs://this/is/bucket")

    def test_has_all_same_scheme(self):
        assert file_utils.has_all_same_scheme(["s3://bucket1", "s3://bucket2"])
        assert file_utils.has_all_same_scheme(["path1", "/path2"])
        assert not file_utils.has_all_same_scheme(["s3://bucket", "path"])

    def test_is_s3_scheme(self):
        assert file_utils.is_s3_scheme("s3")
        assert file_utils.is_s3_scheme("s3a")
        assert file_utils.is_s3_scheme("s3n")
        assert not file_utils.is_s3_scheme("s3x")
        assert not file_utils.is_s3_scheme("aaa")

    def test_get_filesystem(self):
        assert file_utils.get_filesystem("test/path")
        assert file_utils.get_filesystem("s3://bucket")


if __name__ == "__main__":
    absltest.main()
