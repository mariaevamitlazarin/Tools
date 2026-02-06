"""
run_pipeline.py — example of configuring and running the cleaner.

Usage as a script:
    python run_pipeline.py input.csv cleaned.parquet

Or import build_config() and adapt to your own columns.
"""
import sys

from io_utils import read_any, write_any
from cleaner import CleaningConfig, ColumnConfig, DataCleaner

BR_STATES = ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG",
             "PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"]
CITIES = ["São Paulo","Rio de Janeiro","Belo Horizonte","Curitiba","Salvador","Recife"]


def build_config() -> CleaningConfig:
    return CleaningConfig(
        standardize_headers=True,
        drop_empty_rows=True,
        dedup_subset=None,                # full-row exact dedup
        fuzzy_dedup_column="nome",        # catch "Ana Souza" vs "Ana  Souza"
        fuzzy_dedup_threshold=95,
        columns={
            "nome": ColumnConfig(dtype="string", case="title",
                                 regex_replace={r"[-\s]+$": ""}),  # trailing dash/space
            "cidade": ColumnConfig(dtype="string", case="title",
                                   fuzzy_categories=CITIES, fuzzy_threshold=80,
                                   category_map={"Sp": "São Paulo"}),
            "estado": ColumnConfig(dtype="string", case="upper", allowed=BR_STATES),
            "data_de_inscricao": ColumnConfig(
                dtype="datetime",
                date_formats=["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"],
                no_future_dates=True),
            "idade": ColumnConfig(dtype="int", min_value=0, max_value=120,
                                  on_outlier="null"),
            "salario_r": ColumnConfig(dtype="float", min_value=0, on_outlier="flag"),
            "ativo": ColumnConfig(dtype="bool"),
        },
    )


def main(src: str, dst: str):
    df = read_any(src)
    print(f"Loaded {len(df)} rows, {len(df.columns)} cols from {src}")

    cleaner = DataCleaner(build_config())
    clean = cleaner.clean(df)

    write_any(clean, dst)
    print(f"\nWrote {len(clean)} rows to {dst}\n")
    print(cleaner.report_text())

    # also persist the audit log next to the output
    report_path = dst.rsplit(".", 1)[0] + "_report.csv"
    cleaner.report().to_csv(report_path, index=False)
    print(f"\nAudit log -> {report_path}")
    return clean, cleaner


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "sample_messy.csv"
    d = sys.argv[2] if len(sys.argv) > 2 else "cleaned.csv"
    main(s, d)
