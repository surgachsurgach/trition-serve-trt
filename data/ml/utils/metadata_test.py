import json
import os
import tempfile

from absl.testing import absltest

from data.ml.utils import metadata


class MetaTest(absltest.TestCase):
    def test_metadata(self):
        metadata1 = metadata.Meta()
        metadata1.add_meta("item", ["a", "b", "c", "d", "e"])
        metadata1.add_meta("category", ["cat", "gory"])

        assert metadata1.get_meta_size("item") == 5
        assert metadata1.get_meta_size("category") == 2

        idx_to_id = metadata1.get_idx_to_id("item")
        assert idx_to_id == {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

        id_to_idx = metadata1.get_id_to_idx("item")
        assert id_to_idx == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

    def test_metadata_size(self):
        metadata1 = metadata.Meta()
        metadata1.add_meta("item", ["a", "b", "c", "d", "e"])
        metadata1.add_meta("category", ["cat", "gory"])

        assert metadata1.get_meta_size("item") == 5
        assert metadata1.get_meta_size("category") == 2

        assert metadata1.get_meta_size("item", lambda x: x == "a") == 1
        assert metadata1.get_meta_size("item", lambda x: x == "z") == 0

    def test_metadata_start_idx(self):
        metadata1 = metadata.Meta()
        metadata1.add_meta("item", ["a", "b", "c", "d", "e"])
        metadata1.add_meta("category", ["cat", "gory"])

        assert metadata1.get_start_idx("item", lambda x: x.startswith("c")) == 2
        assert metadata1.get_start_idx("item", lambda x: x.startswith("z")) is None

    def test_metadata_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            metadata1 = metadata.Meta()
            metadata1.add_meta("item", ["a", "b", "c", "d", "e"])
            metadata1.add_meta("category", ["cat", "gory"])

            metadata1.save(tmpdirname)

            with open(os.path.join(tmpdirname, metadata.DEFAULT_FILENAME), encoding="utf-8") as jsonfile:
                m = json.load(jsonfile)
            assert m == metadata1._metadata  # pylint: disable=protected-access

            metadata2 = metadata.Meta.from_json(m)
            assert metadata1 == metadata2

            metadata3 = metadata.Meta.load(tmpdirname)
            assert metadata1 == metadata3

    def test_split_item_id(self):
        meta1 = {"item_id": ["1/a", "1/b", "2/c", "2/d", "3/e"]}
        metadata1 = metadata.Meta(metadata=meta1)

        assert metadata1.get_meta("item_id") == ["a", "b", "c", "d", "e"]
        assert metadata1.get_meta("item_id_prefix") == ["1", "1", "2", "2", "3"]

        meta2 = {"item_id": ["a", "b", "c", "d", "e"]}
        metadata2 = metadata.Meta(metadata=meta2)

        assert metadata2.get_meta("item_id") == ["a", "b", "c", "d", "e"]
        assert metadata2.get_meta("item_id_prefix") == ["", "", "", "", ""]


if __name__ == "__main__":
    absltest.main()
