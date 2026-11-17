"""
Unified Multi-Criteria ML Consensus Scoring & Lead Prioritization Engine.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

FEATURES = ["GNINA_Vina", "GNINA_CNN", "MOE_Induced", "ADME", "Toxicology", "PASS"]
INVERT_COLS = ["GNINA_Vina", "MOE_Induced"]

SUPERVISED_WEIGHTS = {
    "GNINA_Vina": 0.25,
    "GNINA_CNN": 0.20,
    "MOE_Induced": 0.20,
    "ADME": 0.12,
    "Toxicology": 0.13,
    "PASS": 0.10,
}


def normalize_if_raw(df: pd.DataFrame, already_normalized: bool = True) -> pd.DataFrame:
    """Applies directional sign inversion & receptor-subgroup Min-Max scaling if raw."""
    if already_normalized:
        return df.copy()

    df_norm = df.copy()
    for col in INVERT_COLS:
        df_norm[col] = -1.0 * df_norm[col].astype(float)

    def _scale(group: pd.DataFrame) -> pd.DataFrame:
        for col in FEATURES:
            c_min, c_max = group[col].min(), group[col].max()
            group[col] = 100.0 if np.isclose(c_max, c_min) else ((group[col] - c_min) / (c_max - c_min)) * 100.0
        return group

    df_norm = df_norm.groupby("Receptor_ID", group_keys=False).apply(_scale)
    df_norm[FEATURES] = df_norm[FEATURES].round(2)
    return df_norm


def run_consensus_scoring(df: pd.DataFrame, top_k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Tier 1 (UCM) and Tier 2 (SCM) scores and extracts Top-K leads per receptor."""
    df_master = df.copy()

    # Tier 1: Unsupervised Consensus Method (Equal 16.67% weight)
    df_master["Unsupervised_Score"] = np.round(df_master[FEATURES].mean(axis=1), 2)

    # Tier 2: Supervised Consensus Method (65% 3D Structural / 35% 2D ADMET)
    df_master["Supervised_Score"] = np.round(
        sum(df_master[f] * SUPERVISED_WEIGHTS[f] for f in FEATURES), 2
    )

    # Hierarchical Sorting: Primary = Supervised_Score, Tie-Breaker = Unsupervised_Score
    df_master = df_master.sort_values(
        by=["Receptor_ID", "Supervised_Score", "Unsupervised_Score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    df_master["Rank_Num"] = df_master.groupby("Receptor_ID").cumcount() + 1
    df_master["Final_Lead_Rank"] = df_master["Rank_Num"].apply(lambda r: f"Top-{r} Lead")

    df_top_k = df_master[df_master["Rank_Num"] <= top_k].copy()
    return df_master, df_top_k


def run_sensitivity_audit(df_master: pd.DataFrame, perturb: float = 0.10, n_iter: int = 500, seed: int = 42) -> pd.DataFrame:
    """Evaluates Top-1 lead retention & Kendall's tau under +/-10% weight perturbations."""
    rng = np.random.default_rng(seed)
    base_w = np.array([SUPERVISED_WEIGHTS[f] for f in FEATURES])
    audit_rows = []

    for rec, group in df_master.groupby("Receptor_ID"):
        base_order = group["Unique_ID"].tolist()
        top1_id = base_order[0]
        retained = 0
        taus = []

        for _ in range(n_iter):
            w_pert = base_w * rng.uniform(1.0 - perturb, 1.0 + perturb, size=len(base_w))
            w_pert /= w_pert.sum()
            scores = sum(group[f] * w for f, w in zip(FEATURES, w_pert))
            pert_order = group.assign(_s=scores).sort_values("_s", ascending=False)["Unique_ID"].tolist()

            if pert_order[0] == top1_id:
                retained += 1
            rank_idx = {uid: i for i, uid in enumerate(pert_order)}
            tau, _ = kendalltau(range(len(base_order)), [rank_idx[u] for u in base_order])
            taus.append(tau)

        audit_rows.append({
            "Receptor_ID": rec,
            "Baseline_Top1_Lead": top1_id,
            "Top1_Retention_%": round((retained / n_iter) * 100, 2),
            "Mean_Kendall_Tau": round(float(np.nanmean(taus)), 4),
        })
    return pd.DataFrame(audit_rows)


def main():
    parser = argparse.ArgumentParser(description="ML Consensus Lead Scoring Framework")
    parser.add_argument("--input", type=str, default="data/processed/normalized_36_hits.csv")
    parser.add_argument("--out-dir", type=str, default="results/tables/")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--raw-input", action="store_true", help="Set if input requires raw Min-Max normalization")
    parser.add_argument("--run-sensitivity", action="store_true", help="Run +/-10% Monte Carlo weight audit")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(args.input)
    df_norm = normalize_if_raw(df_in, already_normalized=not args.raw_input)
    df_master, df_top_k = run_consensus_scoring(df_norm, top_k=args.top_k)

    df_master.to_csv(out_path / "Master_36_Hits_Consensus_Rankings.csv", index=False)
    df_top_k.to_csv(out_path / "Curated_Top3_MD_Simulation_Candidates.csv", index=False)
    print(f"[INFO] Saved master rankings and Top-{args.top_k} prioritized leads to {out_path}")

    if args.run_sensitivity:
        df_audit = run_sensitivity_audit(df_master)
        df_audit.to_csv(out_path / "Weight_Sensitivity_Audit.csv", index=False)
        print(f"[INFO] Saved +/-10% sensitivity audit to {out_path / 'Weight_Sensitivity_Audit.csv'}")


if __name__ == "__main__":
    main()
