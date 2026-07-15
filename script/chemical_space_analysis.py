
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, QED

warnings.filterwarnings("ignore")


# ============================================================
# 1. Utility functions
# ============================================================

def find_smiles_col(df):
    """Automatically find a SMILES-like column."""
    candidate_cols = [
        "canonical_smiles",
        "std_smiles",
        "smiles",
        "SMILES",
        "mol_smiles",
        "compound_smiles",
        "clean_smiles",
        "murcko_scaffold",
        "murcko_scaffold_std",
        "scaffold_smiles",
    ]
    for col in candidate_cols:
        if col in df.columns:
            return col
    raise ValueError(f"No SMILES column found. Available columns: {df.columns.tolist()}")


def safe_read_csv(path):
    """Read CSV with fallback encodings."""
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def extract_scaffold_number(path):
    """Extract scaffold number from file name like df_results1_positive.csv."""
    m = re.search(r"df_results(\d+)_positive", path.name)
    if m:
        return int(m.group(1))
    return None


def mol_from_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def mol_to_canonical_smiles(mol):
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return np.nan


def calc_descriptors(mol):
    try:
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
            "Fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "RingCount": Lipinski.RingCount(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "QED": QED.qed(mol),
        }
    except Exception:
        return {
            "MW": np.nan,
            "LogP": np.nan,
            "TPSA": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "RotB": np.nan,
            "Fsp3": np.nan,
            "RingCount": np.nan,
            "HeavyAtomCount": np.nan,
            "QED": np.nan,
        }


def mol_to_morgan_bitvect(mol, radius=2, n_bits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def bitvect_to_array(fp):
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    # ============================================================
    # 2. Path settings
    # ============================================================

    base_dir = Path(
        r""
    )

    approved_csv = base_dir / "approved_drugs.csv"

    positive_files = [
        base_dir / f"df_results{i}_positive.csv"
        for i in range(1, 11)
    ]

    out_dir = base_dir / "Figure1_10_scaffolds_PCA_UMAP_no_TMAP"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", out_dir)

    # ============================================================
    # 3. Global plotting style
    # ============================================================

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["figure.dpi"] = 300

    sns.set_theme(style="white", context="paper")

    def savefig(fig, name):
        """Save figure as PDF and PNG."""
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=600, bbox_inches="tight")

    # ============================================================
    # 4. Read approved drugs and 10 scaffold files
    # ============================================================

    if not approved_csv.exists():
        raise FileNotFoundError(f"approved_drugs.csv not found: {approved_csv}")

    df_app = safe_read_csv(approved_csv)
    app_smiles_col = find_smiles_col(df_app)

    app = df_app.copy()
    app["dataset_type"] = "Approved drugs"
    app["scaffold_group"] = "Approved drugs"
    app["scaffold_id"] = np.nan
    app["scaffold_label"] = "Approved"
    app["smiles_raw"] = app[app_smiles_col]

    if "pref_name" in app.columns:
        app["compound_name"] = app["pref_name"].astype(str)
    elif "molregno" in app.columns:
        app["compound_name"] = app["molregno"].astype(str)
    else:
        app["compound_name"] = "Approved drug"

    print("Approved drugs:", app.shape, "SMILES column:", app_smiles_col)

    # ---------------- scaffold positive hits ----------------

    pos_list = []

    for f in positive_files:
        if not f.exists():
            print(f"Warning: missing file: {f}")
            continue

        df = safe_read_csv(f)
        smiles_col = find_smiles_col(df)
        scaffold_no = extract_scaffold_number(f)

        tmp = df.copy()
        tmp["dataset_type"] = "Candidate positives"
        tmp["scaffold_id"] = scaffold_no
        tmp["scaffold_label"] = f"S{scaffold_no}"
        tmp["scaffold_group"] = f"S{scaffold_no}"
        tmp["source_file"] = f.name
        tmp["smiles_raw"] = tmp[smiles_col]

        if "molregno" in tmp.columns:
            tmp["compound_name"] = tmp["molregno"].astype(str)
        elif "pref_name" in tmp.columns:
            tmp["compound_name"] = tmp["pref_name"].astype(str)
        else:
            tmp["compound_name"] = tmp["scaffold_label"] + " compound"

        pos_list.append(tmp)
        print(f"Loaded {f.name}: {tmp.shape}, SMILES column: {smiles_col}")

    if len(pos_list) == 0:
        raise ValueError("No positive scaffold files were loaded.")

    pos_all = pd.concat(pos_list, ignore_index=True, sort=False)

    combined_df = pd.concat([app, pos_all], ignore_index=True, sort=False)
    combined_df = combined_df.dropna(subset=["smiles_raw"]).copy()
    combined_df["smiles_raw"] = combined_df["smiles_raw"].astype(str)

    print("Combined raw shape:", combined_df.shape)
    print(combined_df["scaffold_group"].value_counts())

    # ============================================================
    # 5. RDKit cleaning, descriptors and fingerprints
    # ============================================================

    combined_df["mol"] = combined_df["smiles_raw"].apply(mol_from_smiles)
    combined_df = combined_df[combined_df["mol"].notna()].copy()
    combined_df["canonical_smiles_clean"] = combined_df["mol"].apply(mol_to_canonical_smiles)

    combined_df = combined_df.drop_duplicates(
        subset=["scaffold_group", "canonical_smiles_clean"]
    ).reset_index(drop=True)

    print("After RDKit cleaning:", combined_df.shape)
    print(combined_df["scaffold_group"].value_counts())

    # Descriptors
    desc_df = pd.DataFrame(combined_df["mol"].apply(calc_descriptors).tolist())
    combined_df = pd.concat([combined_df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)

    # Morgan fingerprints
    combined_df["fp"] = combined_df["mol"].apply(
        lambda m: mol_to_morgan_bitvect(m, radius=2, n_bits=2048)
    )
    fp_array = np.vstack(combined_df["fp"].apply(bitvect_to_array).values)

    print("Fingerprint matrix:", fp_array.shape)

    combined_df.drop(columns=["mol", "fp"], errors="ignore").to_csv(
        out_dir / "combined_approved_and_10_scaffolds_cleaned.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # 6. Optional downsampling for visualization
    # ============================================================

    MAX_APPROVED_FOR_PLOT = None      
    MAX_PER_SCAFFOLD_FOR_PLOT = None  
    RANDOM_SEED = 42

    plot_index = []

    for group, g in combined_df.groupby("scaffold_group"):
        if group == "Approved drugs":
            max_n = MAX_APPROVED_FOR_PLOT
        else:
            max_n = MAX_PER_SCAFFOLD_FOR_PLOT

        if max_n is not None and len(g) > max_n:
            plot_index.extend(g.sample(max_n, random_state=RANDOM_SEED).index.tolist())
        else:
            plot_index.extend(g.index.tolist())

    plot_index = sorted(plot_index)
    plot_df_base = combined_df.loc[plot_index].copy().reset_index(drop=True)
    fp_array_plot = fp_array[plot_index]

    print("Plot data shape:", plot_df_base.shape)
    print(plot_df_base["scaffold_group"].value_counts())

    # ============================================================
    # 7. Color palette
    # ============================================================

    palette = {
        "Approved drugs": "#0A0A0A",
        "S1": "#B83C35",
        "S2": "#4575B4",
        "S3": "#EBA243FF",
        "S4": "#90F163",
        "S5": "#984EA3",
        "S6": "#66C2A5",
        "S7": "#E7298A",
        "S8": "#A6761D",
        "S9": "#1F78B4",
        "S10": "#B2DF8A",
    }

    scaffold_groups = [f"S{i}" for i in range(1, 11)]

    # ============================================================
    # 8. PCA of physicochemical descriptors
    # ============================================================

    desc_cols = [
        "MW", "LogP", "TPSA", "HBD", "HBA", "RotB",
        "Fsp3", "RingCount", "HeavyAtomCount", "QED"
    ]

    pca_df = plot_df_base.copy()
    X_desc = pca_df[desc_cols].copy()
    X_desc = SimpleImputer(strategy="median").fit_transform(X_desc)
    X_desc_scaled = StandardScaler().fit_transform(X_desc)

    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(X_desc_scaled)

    pca_df["PC1"] = pc[:, 0]
    pca_df["PC2"] = pc[:, 1]

    explained = pca.explained_variance_ratio_ * 100
    loadings = pd.DataFrame(
        pca.components_.T,
        index=desc_cols,
        columns=["PC1", "PC2"]
    )

    pca_df.drop(columns=["mol", "fp"], errors="ignore").to_csv(
        out_dir / "PCA_coordinates_10_scaffolds_vs_approved.csv",
        index=False,
        encoding="utf-8-sig"
    )
    loadings.to_csv(out_dir / "PCA_loadings_10_scaffolds_vs_approved.csv", encoding="utf-8-sig")

    print("PCA explained variance:", explained)
    print(loadings)

    # ============================================================
    # 9. Clean PCA plot: no variable labels, no in-plot labels
    # ============================================================

    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    # Approved drugs: smaller and light grey
    sub = pca_df[pca_df["scaffold_group"] == "Approved drugs"]
    ax.scatter(
        sub["PC1"], sub["PC2"],
        s=3,
        c=palette["Approved drugs"],
        alpha=0.22,
        linewidths=0,
        label="Approved drugs",
        rasterized=True,
        zorder=1,
    )

    # Scaffold groups: smaller points
    for s in scaffold_groups:
        sub = pca_df[pca_df["scaffold_group"] == s]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["PC1"], sub["PC2"],
            s=3,
            c=palette[s],
            alpha=0.78,
            edgecolors="none",
            linewidths=0,
            label=s,
            rasterized=True,
            zorder=2,
        )

    ax.axhline(0, color="#E0E0E0", lw=0.6, zorder=0)
    ax.axvline(0, color="#E0E0E0", lw=0.6, zorder=0)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title("Physicochemical property space", fontsize=10, fontweight="bold")

    # 图注放右侧，不在图中标注
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handletextpad=0.25,
        borderaxespad=0.0,
        markerscale=1.2,
    )

    sns.despine(ax=ax)
    plt.tight_layout()
    savefig(fig, "Figure1D_PCA_clean_no_labels_no_loadings")
    plt.show()

    # ============================================================
    # 10. Supplementary PCA loading barplot
    # Optional, not for main figure
    # ============================================================

    loading_plot_df = loadings.copy()
    loading_plot_df["descriptor"] = loading_plot_df.index
    loading_plot_df["combined_strength"] = np.sqrt(
        loading_plot_df["PC1"] ** 2 + loading_plot_df["PC2"] ** 2
    )
    loading_plot_df = loading_plot_df.sort_values("combined_strength", ascending=True)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    ax.barh(
        loading_plot_df["descriptor"],
        loading_plot_df["combined_strength"],
        color="#4575B4",
        alpha=0.85
    )

    ax.set_xlabel("Loading strength on PC1–PC2")
    ax.set_ylabel("")
    ax.set_title("PCA descriptor contributions", fontsize=9, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()
    savefig(fig, "Supplementary_PCA_loading_strength_barplot")
    plt.show()

    # ============================================================
    # 11. UMAP based on Morgan fingerprints
    # ============================================================

    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn: conda install -c conda-forge umap-learn")

    umap_model = umap.UMAP(
        n_neighbors=30,
        min_dist=0.12,
        n_components=2,
        metric="jaccard",
        random_state=42,
        low_memory=True,
    )

    umap_xy = umap_model.fit_transform(fp_array_plot)

    umap_df = plot_df_base.copy()
    umap_df["UMAP1"] = umap_xy[:, 0]
    umap_df["UMAP2"] = umap_xy[:, 1]

    umap_df.drop(columns=["mol", "fp"], errors="ignore").to_csv(
        out_dir / "UMAP_coordinates_10_scaffolds_vs_approved.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # 12. UMAP plot: no in-plot scaffold labels, smaller points
    # ============================================================

    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    # Approved drugs background
    sub = umap_df[umap_df["scaffold_group"] == "Approved drugs"]
    ax.scatter(
        sub["UMAP1"], sub["UMAP2"],
        s=2.5,
        c=palette["Approved drugs"],
        alpha=0.20,
        linewidths=0,
        label="Approved drugs",
        rasterized=True,
        zorder=1,
    )

    # Ten scaffold groups
    for s in scaffold_groups:
        sub = umap_df[umap_df["scaffold_group"] == s]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["UMAP1"], sub["UMAP2"],
            s=2.5,
            c=palette[s],
            alpha=0.80,
            edgecolors="none",
            linewidths=0,
            label=s,
            rasterized=True,
            zorder=2,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Morgan fingerprint chemical space", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    # legend 放图右侧
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handletextpad=0.25,
        borderaxespad=0.0,
        markerscale=1.4,
    )

    plt.tight_layout()
    savefig(fig, "Figure1B_UMAP_10_scaffolds_vs_approved_no_labels")
    plt.show()

    # ============================================================
    # 13. UMAP density contour plot, no convex hulls
    # ============================================================

    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    # Approved drugs background
    sub_app = umap_df[umap_df["scaffold_group"] == "Approved drugs"]
    ax.scatter(
        sub_app["UMAP1"], sub_app["UMAP2"],
        s=2.0,
        c=palette["Approved drugs"],
        alpha=0.12,
        linewidths=0,
        label="Approved drugs",
        rasterized=True,
        zorder=1,
    )

    # Density contours for each scaffold
    for s in scaffold_groups:
        sub = umap_df[umap_df["scaffold_group"] == s]
        if len(sub) < 10:
            continue

        try:
            sns.kdeplot(
                data=sub,
                x="UMAP1",
                y="UMAP2",
                levels=4,
                color=palette[s],
                linewidths=0.9,
                alpha=0.75,
                fill=False,
                thresh=0.08,
                ax=ax,
                zorder=2,
            )
        except Exception:
            pass

    # Scaffold points
    for s in scaffold_groups:
        sub = umap_df[umap_df["scaffold_group"] == s]
        if len(sub) == 0:
            continue

        ax.scatter(
            sub["UMAP1"], sub["UMAP2"],
            s=2.5,
            c=palette[s],
            alpha=0.65,
            edgecolors="none",
            linewidths=0,
            label=s,
            rasterized=True,
            zorder=3,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Scaffold density in Morgan fingerprint space", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handletextpad=0.25,
        borderaxespad=0.0,
        markerscale=1.4,
    )

    plt.tight_layout()
    savefig(fig, "Figure1C_UMAP_density_contour_no_hulls")
    plt.show()

    # ============================================================
    # 14. UMAP scaffold faceting plot
    # ============================================================

    n_rows = 2
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 3.8), sharex=True, sharey=True)
    axes = axes.flatten()

    sub_app = umap_df[umap_df["scaffold_group"] == "Approved drugs"]

    for i, s in enumerate(scaffold_groups):
        ax = axes[i]

        # Approved drugs background
        ax.scatter(
            sub_app["UMAP1"], sub_app["UMAP2"],
            s=1.2,
            c=palette["Approved drugs"],
            alpha=0.08,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        # Current scaffold
        sub = umap_df[umap_df["scaffold_group"] == s]
        ax.scatter(
            sub["UMAP1"], sub["UMAP2"],
            s=3.0,
            c=palette[s],
            alpha=0.75,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        # Optional density contour for this scaffold
        if len(sub) >= 10:
            try:
                sns.kdeplot(
                    data=sub,
                    x="UMAP1",
                    y="UMAP2",
                    levels=4,
                    color=palette[s],
                    linewidths=0.8,
                    alpha=0.85,
                    fill=False,
                    thresh=0.08,
                    ax=ax,
                    zorder=3,
                )
            except Exception:
                pass

        ax.set_title(s, fontsize=9, fontweight="bold", color=palette[s])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        sns.despine(ax=ax, left=True, bottom=True)

    # Remove unused axes if any
    for j in range(len(scaffold_groups), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Scaffold-specific UMAP neighborhoods", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, "Figure1D_UMAP_scaffold_faceting")
    plt.show()

    # ============================================================
    # 15. Quantitative UMAP cluster statistics
    # ============================================================

    cluster_stats = []

    approved_umap = umap_df[umap_df["scaffold_group"] == "Approved drugs"][["UMAP1", "UMAP2"]].values
    approved_center = np.nanmedian(approved_umap, axis=0)

    for s in scaffold_groups:
        sub = umap_df[umap_df["scaffold_group"] == s]
        if len(sub) == 0:
            continue

        coords = sub[["UMAP1", "UMAP2"]].values
        center = np.nanmedian(coords, axis=0)

        dist_to_approved_center = np.linalg.norm(center - approved_center)
        within_dispersion = np.nanmedian(np.linalg.norm(coords - center, axis=1))

        cluster_stats.append({
            "scaffold_group": s,
            "n_compounds": len(sub),
            "UMAP_center_x": center[0],
            "UMAP_center_y": center[1],
            "distance_to_approved_center": dist_to_approved_center,
            "within_scaffold_dispersion": within_dispersion,
        })

    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df.to_csv(
        out_dir / "UMAP_cluster_statistics_10_scaffolds.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(cluster_stats_df)

    # ============================================================
    # 16. Summary table for manuscript
    # No nearest approved-drug Tanimoto similarity
    # ============================================================

    summary_rows = []

    for scaf in scaffold_groups:
        sub = plot_df_base[plot_df_base["scaffold_group"] == scaf]
        sub_umap = umap_df[umap_df["scaffold_group"] == scaf]

        if len(sub) == 0:
            continue

        if len(sub_umap) > 0:
            umap_center_x = sub_umap["UMAP1"].median()
            umap_center_y = sub_umap["UMAP2"].median()
            umap_dispersion = np.median(
                np.sqrt(
                    (sub_umap["UMAP1"] - umap_center_x) ** 2 +
                    (sub_umap["UMAP2"] - umap_center_y) ** 2
                )
            )
        else:
            umap_center_x = np.nan
            umap_center_y = np.nan
            umap_dispersion = np.nan

        summary_rows.append({
            "scaffold_group": scaf,
            "n_compounds": len(sub),
            "MW_median": sub["MW"].median(),
            "MW_mean": sub["MW"].mean(),
            "LogP_median": sub["LogP"].median(),
            "TPSA_median": sub["TPSA"].median(),
            "HBD_median": sub["HBD"].median(),
            "HBA_median": sub["HBA"].median(),
            "RotB_median": sub["RotB"].median(),
            "Fsp3_median": sub["Fsp3"].median(),
            "RingCount_median": sub["RingCount"].median(),
            "HeavyAtomCount_median": sub["HeavyAtomCount"].median(),
            "QED_median": sub["QED"].median(),
            "UMAP_center_x": umap_center_x,
            "UMAP_center_y": umap_center_y,
            "UMAP_within_scaffold_dispersion": umap_dispersion,
        })

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(
        out_dir / "Figure1_scaffold_summary_statistics_no_Tanimoto.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nScaffold summary statistics:")
    print(summary_table)

    print("\nAll analyses finished.")
    print("Figures and tables saved to:", out_dir)

    print(summary_table)

    print("\nAll analyses finished.")
    print("Figures and tables saved to:", out_dir)


if __name__ == "__main__":
    main()
