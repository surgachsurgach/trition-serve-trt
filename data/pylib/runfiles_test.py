from data.pylib import runfiles


def test_rlocate():
    path = runfiles.Runfiles.rlocate("ridi/data/pylib/testdata/test.txt")
    assert path is not None

    with open(path, encoding="utf-8") as f:
        assert f.read() == "rich imagination and deep insight\n"


def test_locate():
    path = runfiles.Runfiles.locate("data/pylib/testdata/test.txt")
    assert path is not None

    with open(path, encoding="utf-8") as f:
        assert f.read() == "rich imagination and deep insight\n"


def test_rlocate_dirname():
    path = runfiles.Runfiles.rlocate_dirname("ridi/data/pylib/testdata/test.txt")
    assert path is not None
    assert path.endswith("ridi/data/pylib/testdata")


def test_locate_dirname():
    path = runfiles.Runfiles.locate_dirname("data/pylib/testdata/test.txt")
    assert path is not None
    assert path.endswith("ridi/data/pylib/testdata")
