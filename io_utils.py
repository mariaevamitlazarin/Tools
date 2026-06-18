"""
io_utils.py — format-agnostic reading and writing.

Supports CSV, Excel, JSON, Parquet, and SQL (database/API-dumped tables).
Heavy/optional deps (pyarrow, sqlalchemy) are imported lazily so the module
still works for CSV/Excel/JSON when they're not installed.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# Values that should be treated as missing on read, across messy real-world data.
DEFAULT_NA_VALUES = [
    "", " ", "-", "--", "---", "n/a", "N/A", "na", "NA", "NaN", "nan",
    "null", "NULL", "None", "none", "#N/A", "#NA", "?", ".", "<NA>",
    "missing", "MISSING", "desconhecido", "sem informacao", "sem informação",
]


def read_any(
    source: str | Path | io.IOBase,
    fmt: Optional[str] = None,
    *,
    sheet_name: Any = 0,
    encoding: Optional[str] = None,
    sql: Optional[str] = None,
    con: Any = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a dataset from any supported format into a DataFrame.

    fmt is auto-detected from the file suffix when not given. For SQL,
    pass either (sql + con) or a connection string in `source`.
    """
    # --- SQL path -------------------------------------------------------
    if fmt == "sql" or sql is not None:
        if con is None:
            from sqlalchemy import create_engine  # lazy
            con = create_engine(str(source))
        query = sql if sql is not None else f"SELECT * FROM {source}"
        return pd.read_sql(query, con, **kwargs)

    path = Path(str(source))
    suffix = (fmt or path.suffix.lstrip(".")).lower()

    common = dict(na_values=DEFAULT_NA_VALUES, keep_default_na=True)

    if suffix in ("csv", "txt", "tsv"):
        sep = "\t" if suffix == "tsv" else kwargs.pop("sep", None)
        # Robust encoding: try utf-8, fall back to latin-1 (common in BR data).
        for enc in ([encoding] if encoding else ["utf-8", "latin-1", "cp1252"]):
            try:
                return pd.read_csv(
                    source, sep=sep, encoding=enc, engine="python",
                    **common, **kwargs,
                )
            except (UnicodeDecodeError, LookupError):
                continue
        raise UnicodeDecodeError("csv", b"", 0, 1, "could not decode with tried encodings")

    if suffix in ("xlsx", "xls", "xlsm"):
        return pd.read_excel(source, sheet_name=sheet_name, **common, **kwargs)

    if suffix == "json":
        try:
            return pd.read_json(source, **kwargs)
        except ValueError:
            # Fall back to manual load for nested / records-style JSON.
            with open(source, "r", encoding=encoding or "utf-8") as fh:
                data = json.load(fh)
            return pd.json_normalize(data)

    if suffix in ("parquet", "pq"):
        return pd.read_parquet(source, **kwargs)  # needs pyarrow/fastparquet

    raise ValueError(f"Unsupported format: {suffix!r}")


def write_any(df: pd.DataFrame, dest: str | Path, fmt: Optional[str] = None, **kwargs) -> Path:
    """Write a DataFrame to the format implied by the destination suffix."""
    path = Path(str(dest))
    suffix = (fmt or path.suffix.lstrip(".")).lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix in ("csv", "txt"):
        df.to_csv(path, index=False, encoding="utf-8", **kwargs)
    elif suffix in ("xlsx", "xls", "xlsm"):
        df.to_excel(path, index=False, **kwargs)
    elif suffix == "json":
        df.to_json(path, orient="records", force_ascii=False, indent=2, **kwargs)
    elif suffix in ("parquet", "pq"):
        df.to_parquet(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {suffix!r}")
    return path
