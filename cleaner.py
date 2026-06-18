"""
cleaner.py — the configurable cleaning engine.

Design:
  * A `CleaningConfig` declares WHAT to do per column (and globally).
  * `DataCleaner` APPLIES the config step by step, recording every change
    in an audit log so you get a "what changed and why" report.

Every step is independent and optional. Optional libraries (ftfy, rapidfuzz,
pandera) are used when present and silently skipped otherwise.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# ---- optional deps -------------------------------------------------------
try:
    import ftfy  # fixes mojibake like "SÃ£o Paulo" -> "São Paulo"
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

try:
    from rapidfuzz import fuzz, process  # fuzzy dedup / category mapping
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# =========================================================================
# Configuration
# =========================================================================
@dataclass
class ColumnConfig:
    """How to treat a single column."""
    dtype: Optional[str] = None          # "string","int","float","datetime","category","bool"
    strip: bool = True                   # trim whitespace + collapse internal runs
    case: Optional[str] = None           # "lower","upper","title", None
    fix_encoding: bool = True            # repair mojibake (uses ftfy if available)
    deaccent: bool = False               # strip accents (for matching keys)
    category_map: dict[str, str] = field(default_factory=dict)  # exact remaps, e.g. {"SP":"São Paulo"}
    fuzzy_categories: Optional[list[str]] = None  # canonical values to snap to
    fuzzy_threshold: int = 88            # 0-100, higher = stricter
    date_formats: Optional[list[str]] = None      # hints for parsing dates
    min_value: Optional[float] = None    # clamp/flag below this
    max_value: Optional[float] = None    # clamp/flag above this
    no_future_dates: bool = False        # flag dates after "now"
    allowed: Optional[list[Any]] = None  # whitelist of valid values
    regex_replace: dict[str, str] = field(default_factory=dict)  # {pattern: repl}
    fillna: Any = None                   # value to impute missing with
    on_outlier: str = "flag"             # "flag" | "clip" | "null"


@dataclass
class CleaningConfig:
    columns: dict[str, ColumnConfig] = field(default_factory=dict)
    # global behaviours
    drop_exact_duplicates: bool = True
    dedup_subset: Optional[list[str]] = None      # columns that define identity
    fuzzy_dedup_column: Optional[str] = None       # near-duplicate detection on this col
    fuzzy_dedup_threshold: int = 92
    standardize_headers: bool = True               # snake_case, deaccent, trim
    drop_empty_rows: bool = True
    drop_empty_cols: bool = False


# =========================================================================
# Small helpers
# =========================================================================
def _deaccent(s: str) -> str:
    nf = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nf if not unicodedata.combining(c))


def _to_snake(name: str) -> str:
    s = _deaccent(str(name)).strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


_WS = re.compile(r"\s+")


# =========================================================================
# Engine
# =========================================================================
class DataCleaner:
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.log: list[dict] = []      # audit trail
        self.flags: dict[str, list] = {}  # row-level quality flags by reason

    # -- logging -----------------------------------------------------------
    def _record(self, step: str, column: Optional[str], detail: str, n: int):
        if n:
            self.log.append({
                "step": step, "column": column, "detail": detail,
                "rows_affected": int(n),
            })

    def _flag(self, mask: pd.Series, reason: str):
        idx = mask[mask].index.tolist()
        if idx:
            self.flags.setdefault(reason, []).extend(idx)

    # -- main entry --------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        start_rows = len(df)

        if self.config.standardize_headers:
            new_cols = {c: _to_snake(c) for c in df.columns}
            renamed = {k: v for k, v in new_cols.items() if k != v}
            if renamed:
                df = df.rename(columns=new_cols)
                self._record("headers", None, f"renamed {len(renamed)} columns to snake_case", len(renamed))

        if self.config.drop_empty_cols:
            empty = [c for c in df.columns if df[c].isna().all()]
            if empty:
                df = df.drop(columns=empty)
                self._record("drop_empty_cols", None, f"dropped all-null columns: {empty}", len(empty))

        # per-column cleaning
        for col, cfg in self.config.columns.items():
            if col not in df.columns:
                continue
            df[col] = self._clean_column(df, col, cfg)

        if self.config.drop_empty_rows:
            mask = df.isna().all(axis=1)
            if mask.any():
                df = df[~mask]
                self._record("drop_empty_rows", None, "removed fully-empty rows", mask.sum())

        df = self._dedupe(df)

        # attach quality flags as columns
        if self.flags:
            df["_quality_flags"] = ""
            for reason, idxs in self.flags.items():
                valid = [i for i in idxs if i in df.index]
                df.loc[valid, "_quality_flags"] = (
                    df.loc[valid, "_quality_flags"].str.cat([reason] * len(valid), sep=";").str.strip(";")
                )

        self._record("summary", None, f"rows in={start_rows} out={len(df)}", abs(start_rows - len(df)))
        return df.reset_index(drop=True)

    # -- per column --------------------------------------------------------
    def _clean_column(self, df: pd.DataFrame, col: str, cfg: ColumnConfig) -> pd.Series:
        s = df[col]

        # text normalization (only for object/string columns)
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            s = self._clean_text(s, col, cfg)

        # explicit category remap (exact)
        if cfg.category_map:
            before = s.copy()
            s = s.replace(cfg.category_map)
            self._record("category_map", col, f"applied {len(cfg.category_map)} remaps", (before != s).sum())

        # regex replacements
        for pat, repl in cfg.regex_replace.items():
            before = s.copy()
            s = s.astype("string").str.replace(pat, repl, regex=True)
            self._record("regex_replace", col, f"{pat!r}->{repl!r}", (before.fillna("") != s.fillna("")).sum())

        # fuzzy snap to canonical categories
        if cfg.fuzzy_categories and _HAS_RAPIDFUZZ:
            s = self._fuzzy_snap(s, col, cfg)

        # type coercion
        if cfg.dtype:
            s = self._coerce_type(s, col, cfg)

        # range / outlier / future-date checks
        s = self._validate_values(s, col, cfg)

        # imputation last, so it doesn't interfere with validation
        if cfg.fillna is not None:
            n = s.isna().sum()
            if n:
                s = s.fillna(cfg.fillna)
                self._record("fillna", col, f"imputed {n} missing with {cfg.fillna!r}", n)

        return s

    def _clean_text(self, s: pd.Series, col: str, cfg: ColumnConfig) -> pd.Series:
        s = s.astype("string")

        if cfg.fix_encoding:
            if _HAS_FTFY:
                before = s.copy()
                s = s.map(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
                self._record("fix_encoding", col, "ftfy mojibake repair", (before.fillna("") != s.fillna("")).sum())
            else:
                # minimal fallback for the most common latin-1/utf-8 confusion
                def _fix(x):
                    if not isinstance(x, str):
                        return x
                    try:
                        return x.encode("latin-1").decode("utf-8")
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        return x
                before = s.copy()
                s = s.map(_fix)
                self._record("fix_encoding", col, "latin1->utf8 fallback", (before.fillna("") != s.fillna("")).sum())

        if cfg.strip:
            before = s.copy()
            s = s.str.strip().map(lambda x: _WS.sub(" ", x) if isinstance(x, str) else x)
            self._record("strip", col, "trim + collapse whitespace", (before.fillna("") != s.fillna("")).sum())

        if cfg.case:
            fn = {"lower": str.lower, "upper": str.upper, "title": str.title}[cfg.case]
            s = s.map(lambda x: fn(x) if isinstance(x, str) else x)
            self._record("case", col, f"applied {cfg.case}", s.notna().sum())

        if cfg.deaccent:
            s = s.map(lambda x: _deaccent(x) if isinstance(x, str) else x)
            self._record("deaccent", col, "stripped accents", s.notna().sum())

        return s

    def _fuzzy_snap(self, s: pd.Series, col: str, cfg: ColumnConfig) -> pd.Series:
        canon = cfg.fuzzy_categories
        cache: dict[str, str] = {}

        def snap(x):
            if not isinstance(x, str) or not x:
                return x
            if x in cache:
                return cache[x]
            match, score, _ = process.extractOne(x, canon, scorer=fuzz.WRatio)
            cache[x] = match if score >= cfg.fuzzy_threshold else x
            return cache[x]

        before = s.copy()
        s = s.map(snap)
        self._record("fuzzy_categories", col,
                     f"snapped to {len(canon)} canonical values (>= {cfg.fuzzy_threshold})",
                     (before.fillna("") != s.fillna("")).sum())
        return s

    def _coerce_type(self, s: pd.Series, col: str, cfg: ColumnConfig) -> pd.Series:
        dt = cfg.dtype
        before_na = s.isna().sum()
        if dt == "datetime":
            if cfg.date_formats:
                parsed = pd.Series(pd.NaT, index=s.index)
                for fmt in cfg.date_formats:
                    mask = parsed.isna() & s.notna()
                    parsed[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
                # final permissive pass for anything still unparsed
                mask = parsed.isna() & s.notna()
                parsed[mask] = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)
                s = parsed
            else:
                s = pd.to_datetime(s, errors="coerce", dayfirst=True)
        elif dt in ("int", "float"):
            # strip currency symbols, thousands separators, percent, spaces
            cleaned = (s.astype("string")
                         .str.replace(r"[R$€£\s%]", "", regex=True)
                         .str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True)  # pt-BR thousands
                         .str.replace(",", ".", regex=False))
            num = pd.to_numeric(cleaned, errors="coerce")
            s = num.astype("Int64") if dt == "int" else num.astype("float64")
        elif dt == "bool":
            truth = {"true": True, "1": True, "yes": True, "sim": True, "y": True,
                     "false": False, "0": False, "no": False, "nao": False, "não": False, "n": False}
            s = s.astype("string").str.lower().map(truth).astype("boolean")
        elif dt == "category":
            s = s.astype("category")
        elif dt == "string":
            s = s.astype("string")

        coerced_na = s.isna().sum() - before_na
        self._record("coerce_type", col, f"-> {dt} ({coerced_na} unparseable -> NaN)", max(coerced_na, 0))
        return s

    def _validate_values(self, s: pd.Series, col: str, cfg: ColumnConfig) -> pd.Series:
        if cfg.allowed is not None:
            mask = s.notna() & ~s.isin(cfg.allowed)
            self._flag(mask, f"{col}:not_in_allowed")
            self._record("validate_allowed", col, "values outside whitelist", mask.sum())

        is_num = pd.api.types.is_numeric_dtype(s)
        if (cfg.min_value is not None or cfg.max_value is not None) and is_num:
            low = s < cfg.min_value if cfg.min_value is not None else pd.Series(False, index=s.index)
            high = s > cfg.max_value if cfg.max_value is not None else pd.Series(False, index=s.index)
            bad = (low | high).fillna(False)
            if bad.any():
                self._flag(bad, f"{col}:out_of_range")
                if cfg.on_outlier == "clip":
                    s = s.clip(lower=cfg.min_value, upper=cfg.max_value)
                    self._record("outlier_clip", col, "clipped to [min,max]", bad.sum())
                elif cfg.on_outlier == "null":
                    s = s.mask(bad)
                    self._record("outlier_null", col, "nulled out-of-range", bad.sum())
                else:
                    self._record("outlier_flag", col, "flagged out-of-range", bad.sum())

        if cfg.no_future_dates and pd.api.types.is_datetime64_any_dtype(s):
            future = (s > pd.Timestamp(datetime.now())).fillna(False)
            if future.any():
                self._flag(future, f"{col}:future_date")
                self._record("future_date", col, "flagged future dates", future.sum())

        return s

    def _dedupe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.drop_exact_duplicates:
            n = df.duplicated(subset=self.config.dedup_subset).sum()
            if n:
                df = df.drop_duplicates(subset=self.config.dedup_subset)
                self._record("dedup_exact", None,
                             f"removed exact dups (subset={self.config.dedup_subset})", n)

        col = self.config.fuzzy_dedup_column
        if col and col in df.columns and _HAS_RAPIDFUZZ:
            keep, seen = [], []
            thr = self.config.fuzzy_dedup_threshold
            for idx, val in df[col].astype("string").items():
                if not isinstance(val, str) or not val:
                    keep.append(idx); continue
                dup = any(fuzz.WRatio(val, s) >= thr for s in seen)
                if dup:
                    continue
                seen.append(val); keep.append(idx)
            removed = len(df) - len(keep)
            if removed:
                df = df.loc[keep]
                self._record("dedup_fuzzy", col, f"removed near-duplicates (>= {thr})", removed)
        return df

    # -- reporting ---------------------------------------------------------
    def report(self) -> pd.DataFrame:
        """Return the audit log as a DataFrame."""
        return pd.DataFrame(self.log, columns=["step", "column", "detail", "rows_affected"])

    def report_text(self) -> str:
        if not self.log:
            return "No changes recorded."
        lines = ["CLEANING REPORT", "=" * 60]
        for r in self.log:
            scope = f"[{r['column']}]" if r["column"] else "[global]"
            lines.append(f"{r['step']:<18} {scope:<22} {r['detail']}  (rows: {r['rows_affected']})")
        if self.flags:
            lines.append("-" * 60)
            lines.append("QUALITY FLAGS (rows kept but marked):")
            for reason, idxs in self.flags.items():
                lines.append(f"  {reason}: {len(set(idxs))} rows")
        return "\n".join(lines)
