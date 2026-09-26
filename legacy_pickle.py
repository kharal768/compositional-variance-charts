"""Unpickle a pandas 0.x DataFrame under a modern pandas.

LSWMD.pkl was written in 2017-2019 with a pandas version that still had the
``pandas.indexes`` package and the ``Int64Index`` / ``Float64Index`` classes.
Both were removed (module path in 0.20, the index classes in 2.0), so
``pd.read_pickle`` raises ``ModuleNotFoundError: No module named
'pandas.indexes'``.

This module maps the old import paths and class names onto their modern
equivalents at unpickling time.  It does not modify the data.

Usage::

    from legacy_pickle import read_legacy_pickle
    df = read_legacy_pickle("LSWMD.pkl")
"""

from __future__ import annotations

import pickle
import sys
import types

import numpy as np
import pandas as pd


def _install_module_aliases() -> None:
    """Register the pre-0.20 module paths as aliases of the modern ones."""
    import pandas.core.indexes as core_indexes
    import pandas.core.indexes.base as base
    import pandas.core.indexes.datetimes as datetimes
    import pandas.core.indexes.multi as multi
    import pandas.core.indexes.range as range_

    aliases = {
        "pandas.indexes": core_indexes,
        "pandas.indexes.base": base,
        "pandas.indexes.range": range_,
        "pandas.indexes.multi": multi,
        "pandas.indexes.datetimes": datetimes,
    }

    # pandas.indexes.numeric held Int64Index / Float64Index / UInt64Index,
    # all removed in pandas 2.0. A plain Index reproduces their behaviour for
    # the purpose of reconstructing a stored index.
    numeric = types.ModuleType("pandas.indexes.numeric")
    numeric.Int64Index = base.Index
    numeric.Float64Index = base.Index
    numeric.UInt64Index = base.Index
    numeric.NumericIndex = base.Index
    aliases["pandas.indexes.numeric"] = numeric

    for name, module in aliases.items():
        sys.modules.setdefault(name, module)


#: Classes removed outright; every one resolves to a modern replacement.
_CLASS_SUBSTITUTIONS = {
    "Int64Index": "Index",
    "Float64Index": "Index",
    "UInt64Index": "Index",
    "NumericIndex": "Index",
}


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        if module.startswith("pandas.sparse"):
            module = module.replace("pandas.sparse", "pandas.core.sparse", 1)
        if module == "pandas.core.index":
            module = "pandas.core.indexes.base"

        if name in _CLASS_SUBSTITUTIONS:
            from pandas.core.indexes.base import Index

            return Index

        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Last resort: look the name up anywhere in pandas or numpy before
            # giving up, so one renamed internal does not abort the whole load.
            for namespace in (pd, np):
                candidate = getattr(namespace, name, None)
                if candidate is not None:
                    return candidate
            raise


def read_legacy_pickle(path: str, encoding: str = "latin1"):
    """Load the frame.

    ``encoding='latin1'`` is required: the file was pickled under Python 2, so
    its byte strings have no encoding attached. Latin-1 is the standard choice
    because it round-trips every byte value 0-255 without loss, which matters
    here --- the wafer maps are stored as raw numpy buffers and a lossy codec
    would silently corrupt them.
    """
    _install_module_aliases()
    with open(path, "rb") as fh:
        return LegacyUnpickler(fh, encoding=encoding).load()
