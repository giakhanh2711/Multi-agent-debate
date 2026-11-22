import statistics
import math
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2.2 Compute the average correctness
# 2.3 Compute 95% confidence intervals
def mean_ci(values):
    values = list(values)
    mean = statistics.fmean(values)
    half_width = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half_width


def summarize_results(input_jsonl: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = Path(input_jsonl).read_text(encoding='utf-8').splitlines()
    data = []
    for line in records:
        data.append(json.loads(line.strip()))

    df = pd.DataFrame(data)

    summaries = []
    for label, group in df.groupby("condition", sort=False):
        n = len(group)
        successes = int(group["is_correct"].sum())
        acc, ci_half_width = mean_ci(group["is_correct"])
        summaries.append({
            "condition": label,
            "n": n,
            "n_correct": successes,
            "acc": acc,
            "ci95": ci_half_width
        })

    summaries = pd.DataFrame(summaries)

    summaries.to_csv(output_dir / "reproduced_summary.csv", index=False)

    x = np.arange(len(summaries))
    means = summaries['acc'].to_numpy()
    lower = np.maximum(0, means - summaries["ci95"].to_numpy())
    upper = np.minimum(1.0, means + summaries["ci95"].to_numpy())

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, means, yerr=np.vstack([lower, upper]), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summaries["condition"], rotation=30, ha="right")
    ax.set_ylabel("Acc")
    ax.set_ylim(0, 1)
    ax.set_title("TriviaQA accuray by debate configuration (95% CIs)")
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_plot.png", dpi=200)

    return summaries



def summarize_payoffs(
    input_jsonl: str,
    strategies: list,
    output_dir: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = Path(input_jsonl).read_text(encoding="utf-8").splitlines()
    data = [json.loads(line) for line in records if line.strip()]
    df = pd.DataFrame(data)

    n = len(strategies)
    strategy_order = {s: i for i, s in enumerate(strategies)}

    def canonical_pair(row):
        s1, s2 = row["strategy1"], row["strategy2"]
        if strategy_order[s1] <= strategy_order[s2]:
            return s1, s2
        return s2, s1

    df[["profile1", "profile2"]] = df.apply(
        lambda row: pd.Series(canonical_pair(row)),
        axis=1,
    )

    matrix = np.zeros((n, n, 2), dtype=float)
    ci_rows = []

    grouped = df.groupby(["profile1", "profile2"], sort=False)

    for (s1, s2), group in grouped:
        question_means = group.groupby("question_id")["payoff"].mean()

        mean = float(question_means.mean())

        rng = np.random.default_rng(
            42 + strategy_order[s1] * 10 + strategy_order[s2]
        )

        q_values = question_means.to_numpy()
        bootstrap_means = [
            rng.choice(q_values, size=len(q_values), replace=True).mean()
            for _ in range(5000)
        ]

        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

        i, j = strategy_order[s1], strategy_order[s2]

        matrix[i, j, :] = mean
        matrix[j, i, :] = mean

        for a, b in {(s1, s2), (s2, s1)}:
            ci_rows.append({
                "strategy1": a,
                "strategy2": b,
                "mean_payoff": mean,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "n_debates": len(group),
                "n_questions": len(question_means),
            })

    pd.DataFrame(ci_rows).to_csv(
        output_dir / "payoff_ci.csv", index=False
    )

    payoff_df = pd.DataFrame(
        matrix[:, :, 0],
        index=strategies,
        columns=strategies,
    )
    payoff_df.to_csv(output_dir / "payoff_matrix.csv")

    np.save(output_dir / "payoff_matrix.npy", matrix)

    return matrix, pd.DataFrame(ci_rows)


def clustered_bootstrap_ci(
    records,
    metric,
    cluster_key="question_id",
    n_bootstrap=5000,
    seed=42,
):
   
    df = pd.DataFrame(records)

    if df.empty:
        return float("nan"), float("nan")

    cluster_means = (
        df.groupby(cluster_key)[metric]
        .mean()
        .to_numpy()
    )

    if len(cluster_means) < 2:
        mean = float(np.mean(cluster_means))
        return mean, mean

    rng = np.random.default_rng(seed)

    bootstrap_means = np.array([
        rng.choice(
            cluster_means,
            size=len(cluster_means),
            replace=True
        ).mean()
        for _ in range(n_bootstrap)
    ])

    low, high = np.percentile(bootstrap_means, [2.5, 97.5])

    return float(low), float(high)


def summarize_by_groups(
    records,
    group_keys,
    metrics,
    cluster_key="question_id",
    n_bootstrap=5000,
    seed=42,
):
    """
    Generic summary for one or more grouping columns.

    Returns mean and 95% clustered-bootstrap CI for each metric.
    """
    df = pd.DataFrame(records)

    summaries = []

    grouped = df.groupby(group_keys, sort=False)

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(zip(group_keys, group_key))
        row["n_debates"] = len(group)
        row["n_questions"] = group[cluster_key].nunique()

        for metric_index, metric in enumerate(metrics):
            row[f"mean_{metric}"] = float(group[metric].mean())

            ci_low, ci_high = clustered_bootstrap_ci(
                group.to_dict("records"),
                metric,
                cluster_key=cluster_key,
                n_bootstrap=n_bootstrap,
                seed=seed + metric_index,
            )

            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high

        summaries.append(row)

    return pd.DataFrame(summaries)
