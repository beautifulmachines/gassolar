from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("gassolar")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
