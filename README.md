# Data Cleaning & Normalization Pipeline

A reusable, configurable pipeline (pandas/numpy) for cleaning messy datasets:
missing values, inconsistent text, wrong types, duplicates, outliers,
unit/currency inconsistency, and encoding problems. Every change is recorded
in an audit report.

## Files
- `io_utils.py` — read/write CSV, Excel, JSON, Parquet, SQL (one interface).
- `cleaner.py` — the engine: `CleaningConfig`, `ColumnConfig`, `DataCleaner`.
- `run_pipeline.py` — example config + CLI.
- `make_sample.py` — generates a deliberately messy CSV to test against.

## Install
Core works with just pandas + numpy + openpyxl. Turn on the strongest features
with the optional libraries:

```bash
pip install pandas numpy openpyxl          # core (required)
pip install rapidfuzz ftfy pyarrow         # fuzzy match/dedup, mojibake, parquet
pip install pandera sqlalchemy             # schema validation, databases/APIs
```

The code detects what's installed and degrades gracefully:
- no `rapidfuzz` → fuzzy category-snap and fuzzy dedup are skipped (exact still runs)
- no `ftfy` → falls back to a basic latin1→utf8 repair
- no `pyarrow` → parquet read/write unavailable (other formats fine)

## Quick start
```bash
python make_sample.py
python run_pipeline.py sample_messy.csv cleaned.parquet
```

Or in code:
```python
from io_utils import read_any, write_any
from cleaner import CleaningConfig, ColumnConfig, DataCleaner

df = read_any("input.xlsx")                 # or .csv/.json/.parquet
cfg = CleaningConfig(
    fuzzy_dedup_column="name",
    columns={
        "name":  ColumnConfig(dtype="string", case="title"),
        "city":  ColumnConfig(fuzzy_categories=["São Paulo","Rio de Janeiro"]),
        "state": ColumnConfig(case="upper", allowed=["SP","RJ","MG"]),
        "date":  ColumnConfig(dtype="datetime", no_future_dates=True),
        "age":   ColumnConfig(dtype="int", min_value=0, max_value=120, on_outlier="null"),
        "price": ColumnConfig(dtype="float", min_value=0),   # strips R$, ., , %
    },
)
cleaner = DataCleaner(cfg)
clean = cleaner.clean(df)
write_any(clean, "clean.parquet")
print(cleaner.report_text())     # human-readable
cleaner.report().to_csv("audit.csv")   # machine-readable
```

## ColumnConfig options
| option | purpose |
|---|---|
| `dtype` | `string/int/float/datetime/category/bool` — coercion, bad values → NaN |
| `strip`, `case`, `deaccent` | text normalization |
| `fix_encoding` | repair mojibake (ftfy if present) |
| `category_map` | exact remaps, e.g. `{"SP":"São Paulo"}` |
| `fuzzy_categories` + `fuzzy_threshold` | snap variants to canonical list |
| `date_formats` | parsing hints, tried in order, permissive fallback |
| `min_value`/`max_value` + `on_outlier` | range checks: `flag`/`clip`/`null` |
| `no_future_dates` | flag dates after now |
| `allowed` | whitelist; non-members flagged |
| `regex_replace` | `{pattern: replacement}` |
| `fillna` | impute missing |

Global (`CleaningConfig`): header standardization, exact + fuzzy dedup,
empty row/col dropping.

## Output
- Cleaned dataset in any supported format.
- A `_quality_flags` column marking rows kept but suspect (out-of-range,
  future date, not-in-whitelist) — so you decide whether to drop them.
- An audit report (`.report_text()` / `.report()`).

## Scaling to millions of rows
- Parquet I/O + dtype coercion already keep memory down.
- For very large files, read CSV in chunks (`pd.read_csv(..., chunksize=...)`)
  and clean per chunk, or swap the engine to **Polars** — the same
  `ColumnConfig` ideas port over (`.str`, `.cast`, `.fill_null`).
- Fuzzy dedup is O(n²); for large data, block first (e.g. group by city/state)
  and fuzzy-match only within blocks.

## Optional: hard schema validation with pandera
After cleaning, enforce a contract that fails loudly in a scheduled job:
```python
import pandera as pa
schema = pa.DataFrameSchema({
    "age": pa.Column("Int64", pa.Check.in_range(0,120), nullable=True),
    "state": pa.Column("string", pa.Check.isin(["SP","RJ","MG"])),
})
schema.validate(clean, lazy=True)
```
