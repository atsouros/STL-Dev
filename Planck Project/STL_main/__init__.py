import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("STL")
except PackageNotFoundError:
    __version__ = None

    warnings.warn(
        "In development mode: Please run 'pip install -e .' in the root directory of the STL repository to properly install the package, its dependencies and enable proper version tracking.",
        UserWarning,
    )
