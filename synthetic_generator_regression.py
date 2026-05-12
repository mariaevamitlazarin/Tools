"""
Synthetic data generator for testing regression models before real data exists.

Combines:
  - sklearn.make_regression  -> learnable numeric signal + continuous target
  - Faker                    -> realistic-looking metadata columns
Output: a single pandas DataFrame.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.datasets import make_regression


def generate(
    n_rows: int = 1000,
    # --- numeric signal (passed to make_regression) ---
    n_features: int = 8,
    n_informative: int = 5,
    noise: float = 10.0,             # std of gaussian noise on target; higher = harder
    n_targets: int = 1,             # >1 produces multi-output regression
    bias: float = 0.0,              # constant offset added to target
    tail_strength: float = 0.5,     # relative importance of fat-noisy tail of singular values
    target_name: str = "target",
    # --- realistic Faker columns ---
    faker_fields: tuple[str, ...] = ("name", "email", "country", "job", "signup_date"),
    correlate_with_target: bool = True,  # tie one field's distribution to target value
    # --- dirty-data injection ---
    missing_frac: float = 0.0,       # fraction of cells set to NaN (numeric cols)
    duplicate_frac: float = 0.0,     # fraction of rows duplicated
    outlier_frac: float = 0.0,       # fraction of numeric cells blown up
    # --- reproducibility ---
    seed: int = 42,
    locale: str = "en_US",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker(locale)
    Faker.seed(seed)

    X, y = make_regression(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        noise=noise,
        bias=bias,
        tail_strength=tail_strength,
        random_state=seed,
    )

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    if n_targets == 1:
        df[target_name] = y
    else:
        for i in range(n_targets):
            df[f"{target_name}_{i}"] = y[:, i]

    # normalized target (0..1) used for correlation logic below
    y_scalar = y if n_targets == 1 else y[:, 0]
    y_norm = (y_scalar - y_scalar.min()) / (np.ptp(y_scalar) + 1e-9)

    # --- Faker columns ---
    field_makers = {
        "name": fake.name,
        "email": fake.email,
        "country": fake.country,
        "job": fake.job,
        "company": fake.company,
        "phone": fake.phone_number,
        "address": fake.address,
        "signup_date": lambda: fake.date_between("-3y", "today"),
        "city": fake.city,
    }
    for field in faker_fields:
        if field not in field_makers:
            raise ValueError(f"Unknown faker field '{field}'. "
                             f"Available: {sorted(field_makers)}")
        df[field] = [field_makers[field]() for _ in range(n_rows)]

    # tie 'age' to the target so a realistic-looking field carries signal
    if correlate_with_target:
        base = rng.normal(35, 8, n_rows)
        df["age"] = (base + y_norm * 30).round().clip(18, 90).astype(int)

    # --- dirty data ---
    num_cols = [c for c in df.columns if c.startswith("feat_")]
    if missing_frac > 0:
        mask = rng.random(df[num_cols].shape) < missing_frac
        df[num_cols] = df[num_cols].mask(mask)
    if outlier_frac > 0:
        mask = rng.random(df[num_cols].shape) < outlier_frac
        df[num_cols] = df[num_cols] + mask * rng.normal(0, 50, df[num_cols].shape)
    if duplicate_frac > 0:
        n_dup = int(n_rows * duplicate_frac)
        dups = df.sample(n_dup, random_state=seed)
        df = pd.concat([df, dups], ignore_index=True).sample(
            frac=1, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Example: noisy, slightly dirty regression test set
    df = generate(
        n_rows=2000,
        n_features=10,
        n_informative=6,
        noise=25.0,
        target_name="price",
        faker_fields=("name", "email", "country", "signup_date"),
        missing_frac=0.02,
        duplicate_frac=0.01,
        outlier_frac=0.005,
        seed=7,
    )
    print(df.head())
    print("\nshape:", df.shape)
    print("\ntarget stats:\n", df["price"].describe())
