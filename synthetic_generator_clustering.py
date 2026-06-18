"""
Synthetic data generator for testing clustering models before real data exists.

Combines:
  - sklearn.make_blobs  -> well-separated gaussian clusters (default)
    or make_moons / make_circles for non-convex shapes
  - Faker               -> realistic-looking metadata columns
Output: a single pandas DataFrame. The true cluster id is included as
'cluster' for evaluation (ARI, NMI, etc.) but should NOT be fed to the model.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.datasets import make_blobs, make_moons, make_circles


def generate(
    n_rows: int = 1000,
    # --- cluster structure ---
    shape: str = "blobs",            # "blobs" | "moons" | "circles"
    n_clusters: int = 4,             # only used by "blobs"
    n_features: int = 8,             # only used by "blobs"; moons/circles are 2D
    cluster_std: float = 1.0,        # spread; higher = more overlap = harder
    center_box: tuple[float, float] = (-10.0, 10.0),  # blob center range
    noise: float = 0.08,             # only used by moons/circles
    # --- realistic Faker columns ---
    faker_fields: tuple[str, ...] = ("name", "email", "country", "job", "signup_date"),
    correlate_with_cluster: bool = True,  # tie one field's distribution to cluster
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

    if shape == "blobs":
        X, y = make_blobs(
            n_samples=n_rows,
            centers=n_clusters,
            n_features=n_features,
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=seed,
        )
    elif shape == "moons":
        X, y = make_moons(n_samples=n_rows, noise=noise, random_state=seed)
    elif shape == "circles":
        X, y = make_circles(n_samples=n_rows, noise=noise,
                            factor=0.5, random_state=seed)
    else:
        raise ValueError(f"Unknown shape '{shape}'. Use blobs | moons | circles.")

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df["cluster"] = y  # ground truth — for scoring only, not for the model

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

    # tie 'age' to the cluster so a realistic-looking field carries structure
    if correlate_with_cluster:
        base = rng.normal(35, 6, n_rows)
        df["age"] = (base + y * 8).round().clip(18, 90).astype(int)

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
    # Example: 5 gaussian blobs with mild overlap and a little dirt
    df = generate(
        n_rows=1500,
        shape="blobs",
        n_clusters=5,
        n_features=6,
        cluster_std=1.4,
        faker_fields=("name", "email", "country", "signup_date"),
        missing_frac=0.02,
        outlier_frac=0.005,
        seed=7,
    )
    print(df.head())
    print("\nshape:", df.shape)
    print("\ncluster sizes:\n", df["cluster"].value_counts().sort_index())
