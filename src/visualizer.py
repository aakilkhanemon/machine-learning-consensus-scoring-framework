"""
Publication-Grade Pareto Therapeutic Landscape Visualizer (600 DPI PNG + Vector SVG).
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["svg.fonttype"] = "none"


def plot_consensus_landscape(df_top: pd.DataFrame, out_dir: Path, dpi: int = 600) -> None:
    """Renders 600 DPI Pareto scatter plot of Top-3 prioritized leads across all 6 receptors."""
    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=dpi)

    markers = ["o", "s", "^", "D", "X", "P"]
    palette = sns.color_palette("bright", df_top["Receptor_ID"].nunique())

    for i, rec in enumerate(sorted(df_top["Receptor_ID"].unique())):
        sub_df = df_top[df_top["Receptor_ID"] == rec]
        ax.scatter(
            sub_df["Unsupervised_Score"],
            sub_df["Supervised_Score"],
            label=rec,
            s=175,
            alpha=0.9,
            edgecolors="#222222",
            linewidths=1.4,
            marker=markers[i % len(markers)],
            color=palette[i],
            zorder=3,
        )

    # Non-overlapping label offsets for Top-1 leads
    offsets = {
        "Rec-1": (1.4, -0.6),
        "Rec-2": (-15.0, -2.0),
        "Rec-3": (1.4, 0.6),
        "Rec-4": (1.4, -1.6),
        "Rec-5": (-15.5, 1.0),
        "Rec-6": (1.4, 0.8),
    }

    for rec, sub_df in df_top.groupby("Receptor_ID"):
        top_1 = sub_df.sort_values(["Supervised_Score", "Unsupervised_Score"], ascending=False).iloc[0]
        x_off, y_off = offsets.get(rec, (1.2, 0.0))
        ax.text(
            top_1["Unsupervised_Score"] + x_off,
            top_1["Supervised_Score"] + y_off,
            f"{top_1['Compound_Type']}\n({top_1['Unique_ID']} | {rec})",
            fontsize=8.5,
            weight="bold",
            color="#111111",
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="#cccccc", boxstyle="round,pad=0.25"),
            zorder=4,
        )

    ax.set_xlim(df_top["Unsupervised_Score"].min() - 6, 108)
    ax.set_ylim(df_top["Supervised_Score"].min() - 6, 105)
    ax.set_xlabel("Unsupervised Baseline Score (Equal-Weight 16.67% Balance)", fontsize=11, weight="bold", labelpad=10)
    ax.set_ylabel("Supervised Consensus Selection Score (65% Structural / 35% ADMET)", fontsize=11, weight="bold", labelpad=10)
    ax.set_title(
        "Multi-Target Therapeutic Landscape: Machine Learning Consensus Metrics\nTop-3 Prioritized Leads Evaluated Across Six Viral Receptor Pockets",
        fontsize=12.5,
        weight="bold",
        pad=14,
    )

    ax.grid(True, linestyle=":", alpha=0.6, color="#aaaaaa", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Target Receptor", title_fontsize=10.5, fontsize=9.5, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    png_path = out_dir / "Fig3_ML_Consensus_Landscape.png"
    svg_path = out_dir / "Fig3_ML_Consensus_Landscape.svg"
    plt.savefig(png_path, bbox_inches="tight", dpi=dpi)
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Exported 600 DPI PNG and vector SVG to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render Pareto Landscape Figure")
    parser.add_argument("--input", type=str, default="results/tables/Curated_Top3_MD_Simulation_Candidates.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures/")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df_top = pd.read_csv(args.input)
    plot_consensus_landscape(df_top, out_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
