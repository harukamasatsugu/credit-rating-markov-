import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


DEFAULT_TM = np.array([
    [0.90, 0.08, 0.01, 0.005, 0.004, 0.001, 0.000, 0.000],  # AAA
    [0.02, 0.88, 0.07, 0.015, 0.010, 0.004, 0.001, 0.000],  # AA
    [0.01, 0.03, 0.85, 0.080, 0.020, 0.007, 0.002, 0.001],  # A
    [0.002, 0.01, 0.05, 0.820, 0.080, 0.030, 0.005, 0.003], # BBB
    [0.001, 0.002, 0.01, 0.060, 0.780, 0.100, 0.030, 0.017],# BB
    [0.000, 0.001, 0.002, 0.010, 0.080, 0.750, 0.100, 0.057],# B
    [0.000, 0.000, 0.001, 0.005, 0.020, 0.120, 0.600, 0.254],# CCC
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],# D (absorbing)
], dtype=float)


def normalize_rows(tm: np.ndarray) -> np.ndarray:
    row_sums = tm.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("Found a row with sum=0 in transition matrix.")
    return tm / row_sums


def plot_transition_heatmap(df_tm: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(df_tm.values)

    ax.set_xticks(range(len(df_tm.columns)), df_tm.columns)
    ax.set_yticks(range(len(df_tm.index)), df_tm.index)
    ax.set_xlabel("Target Rating")
    ax.set_ylabel("Source Rating")
    ax.set_title("Credit Rating Transition Matrix (1-Year)")

    for i in range(df_tm.shape[0]):
        for j in range(df_tm.shape[1]):
            ax.text(j, i, f"{df_tm.iat[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def simulate_paths(tm: np.ndarray, start_idx: int, years: int, n_paths: int, rng: np.random.Generator):
    n_states = tm.shape[0]
    paths = []
    for _ in range(n_paths):
        cur = start_idx
        path = [cur]
        for _ in range(years):
            cur = rng.choice(np.arange(n_states), p=tm[cur])
            path.append(cur)
        paths.append(path)
    return np.array(paths)  # shape: (n_paths, years+1)


def plot_sample_trajectories(paths: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, path in enumerate(paths):
        ax.plot(path, marker="o", alpha=0.7, label=f"Sample {i+1}")

    ax.set_yticks(range(len(RATINGS)), RATINGS)
    ax.invert_yaxis()
    ax.set_title("Sample Trajectories (Markov Simulation)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Rating")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def project_distribution(tm: np.ndarray, start_dist: np.ndarray, years: int) -> pd.DataFrame:
    dist = start_dist.copy()
    history = [dist]
    for _ in range(years):
        dist = dist @ tm
        history.append(dist)
    return pd.DataFrame(history, columns=RATINGS)


def plot_distribution_area(df_dist: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    df_dist.plot(kind="area", stacked=True, alpha=0.8, ax=ax)
    ax.set_title("Projected Rating Distribution Over Time")
    ax.set_xlabel("Years")
    ax.set_ylabel("Proportion")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default="BBB", choices=RATINGS)
    parser.add_argument("--path-years", type=int, default=20)
    parser.add_argument("--n-paths", type=int, default=5)
    parser.add_argument("--dist-years", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tm = normalize_rows(DEFAULT_TM)
    df_tm = pd.DataFrame(tm, index=RATINGS, columns=RATINGS)

    rng = np.random.default_rng(args.seed)

    # 1) Heatmap
    plot_transition_heatmap(df_tm, outdir / "transition_matrix_heatmap.png")

    # 2) Sample paths
    start_idx = RATINGS.index(args.start)
    paths = simulate_paths(tm, start_idx=start_idx, years=args.path_years, n_paths=args.n_paths, rng=rng)
    plot_sample_trajectories(paths, outdir / "sample_trajectories.png")

    # 3) Distribution projection
    start_dist = np.zeros(len(RATINGS))
    start_dist[start_idx] = 1.0
    df_dist = project_distribution(tm, start_dist=start_dist, years=args.dist_years)
    plot_distribution_area(df_dist, outdir / "distribution_area.png")

    # 4) Cumulative PD
    print(f"--- Cumulative Default Probability from {args.start} ---")
    for year in [1, 5, 10, 20]:
        if year <= args.dist_years:
            print(f"Year {year:2d}: {df_dist.iloc[year]['D']:.4%}")


if __name__ == "__main__":
    main()
