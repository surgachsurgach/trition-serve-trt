"""
Refer:
    https://github.com/bazelbuild/bazel/blob/master/tools/python/runfiles/runfiles.py
"""
import os

from bazel_tools.tools.python.runfiles import runfiles


class Runfiles:

    _RUNFILE = runfiles.Create()

    @classmethod
    def rlocate(cls, path: str) -> str:
        """Locate a file in the runfiles tree.

        Args:
            path: The path to the file, relative to the runfiles root.

        Returns:
            Returns the runtime path of a runfile.
        """
        return cls._RUNFILE.Rlocation(path)

    @classmethod
    def locate(cls, path: str) -> str:
        """Locate a file in the relative to WORKSPACE root.

        Args:
            path: The path to the file, relative to the WORKSPACE root.

        Returns:
            Returns the runtime path of a runfile.
        """
        full_path = os.path.join("ridi", path)
        return cls.rlocate(full_path)

    @classmethod
    def rlocate_dirname(cls, path: str) -> str:
        """Locate a file in the runfiles tree and return its dirname.

        Args:
            path: The path to the file, relative to the runfiles root.

        Returns:
            Returns the runtime dirname of a runfile.
        """
        path = cls.rlocate(path)
        return os.path.dirname(cls.rlocate(path))

    @classmethod
    def locate_dirname(cls, path: str) -> str:
        """Locate a file in the relative to WORKSPACE root and return its dirname.

        Args:
            path: The path to the file, relative to the WORKSPACE root.

        Returns:
            Returns the runtime dirname of a runfile.
        """
        full_path = os.path.join("ridi", path)
        return cls.rlocate_dirname(full_path)
