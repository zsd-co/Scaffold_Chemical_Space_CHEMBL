
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

import chembl_downloader
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS

warnings.filterwarnings("ignore")

SQL_QUERY = """
SELECT 
    md.molregno,
    cs.canonical_smiles,
    md.pref_name,
    md.first_approval,
    md.dosed_ingredient,
    cp.full_mwt,
    cp.alogp AS logp,
    act.standard_value,
    act.standard_type,
    act.standard_relation,
    act.standard_units,
    td.tid,
    td.target_type,
    td.organism,
    td.pref_name AS target_name,
    ass.assay_type,
    ass.confidence_score,
    act.activity_id
FROM 
    molecule_dictionary md
JOIN compound_structures cs ON md.molregno = cs.molregno
JOIN compound_properties cp ON md.molregno = cp.molregno
JOIN activities act ON md.molregno = act.molregno
JOIN assays ass ON act.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
WHERE 
    md.molecule_type = 'Small molecule'
    AND cp.full_mwt > 200 
    AND cp.full_mwt < 1000
"""

skeleton_smiles_list = [
    "O=C1NC2=CC=CC=C2C1",
    "O=C1NC2=CC=CC=C2C13CCCC3",
    "O=C1NC2=CC=CC=C2C13CCC4C3CCC4",
    "O=C1NC2=CC=CC=C2C13CC3",
    "O=C1NC2=CC=CC=C2C13CCCCC3",
    "O=C1NC2=CC=CC=C2C13CC(C=CC=C4)=C4CC3",
    "O=C1NC2=CC=CC=C2C13CC(C=CC=C4)=C4C3",
    "O=C1NC2=CC=CC=C2C13CCC3",
    "O=C1NC2=CC=CC=C2C13C4(CCCC4)CCC3",
    "O=C1NC2=CC=CC=C2C13CCCC4=C3NC5=C4C=CC=C5",
]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def has_benzo_five_membered_ring(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        ri = mol.GetRingInfo()
        all_rings = list(ri.AtomRings())

        benzene_rings = []
        for ring in all_rings:
            if len(ring) == 6:
                atoms = [mol.GetAtomWithIdx(i) for i in ring]
                if all(a.GetIsAromatic() for a in atoms):
                    benzene_rings.append(set(ring))

        if not benzene_rings:
            return False

        for ring in all_rings:
            if len(ring) == 5:
                ring_set = set(ring)
                for benz_set in benzene_rings:
                    if len(benz_set & ring_set) >= 2:
                        return True

        return False
    except Exception:
        return False


def query_chembl(sql: str) -> pd.DataFrame:
    df = chembl_downloader.query(sql)
    return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df = df[
        (df['standard_type'].isin(['IC50', 'Ki', 'Kd', 'EC50'])) &
        (df['standard_value'].notna()) &
        (df['standard_units'] == 'nM') &
        (df['standard_relation'].isin(['=', '~', '>', '<', '>=', '<=']))
    ].copy()

    df = df[
        (df['target_type'] == 'SINGLE PROTEIN') &
        (df['organism'] == 'Homo sapiens') &
        (df['target_name'].notna())
    ].copy()

    df = df[
        (df['assay_type'].isin(['B', 'F', 'A'])) &
        (df['confidence_score'] >= 7)
    ].copy()

    df['pic50'] = -np.log10(df['standard_value'] / 1e9)
    df = df[df['pic50'] >= 5.5].copy()

    df = df[(df['logp'].between(-2, 7))].copy()

    df['is_marketed'] = df['first_approval'].notna().astype(int)

    mol_stats = df.groupby('molregno').agg(
        target_count=('tid', 'nunique'),
        total_measurements=('activity_id', 'nunique')
    ).reset_index()
    df = df.merge(mol_stats, on='molregno', how='left')
    df['target_density'] = np.where(
        df['total_measurements'] > 0,
        df['target_count'] / df['total_measurements'],
        0
    )

    return df


def filter_pains(df: pd.DataFrame) -> pd.DataFrame:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    def is_pains(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return True
            return catalog.HasMatch(mol)
        except Exception:
            return True

    df['is_pains'] = df['canonical_smiles'].apply(is_pains)
    df_filtered = df[~df['is_pains']].copy()
    return df_filtered


# ===================== molecular aggregation function =====================
def aggregate_unique_molecules(df):
    if df.empty:
        return df
    agg_cols = {
        "canonical_smiles": "first",
        "pref_name": "first",
        "first_approval": "first",
        "dosed_ingredient": "first",
        "full_mwt": "first",
        "logp": "first",
        "pic50": "mean",
        "target_type": "first",
        "organism": "first",
        "target_name": lambda x: ', '.join(x.dropna().unique()),
        "assay_type": "first",
        "confidence_score": "first",
        "is_marketed": "first",
        "target_count": "first",
        "total_measurements": "first",
        "target_density": "first",
    }
    df_agg = df.groupby("molregno", as_index=False).agg(agg_cols)
    return df_agg
# =======================================================


def get_murcko_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def assign_scaffold_label(df: pd.DataFrame) -> pd.DataFrame:
    df['murcko_scaffold'] = df['canonical_smiles'].apply(get_murcko_smiles)

    cond1 = df['full_mwt'] > 500
    cond2 = df['logp'] > 5
    cond3 = df['target_count'] > 5

    bad_condition_count = cond1.astype(int) + cond2.astype(int) + cond3.astype(int)
    df['scaffold_label'] = (bad_condition_count < 2).astype(int)
    return df


def generalize_aliphatic_atoms(mol):
    if mol is None:
        return None
    mol_generic = Chem.Mol(mol)
    for atom in mol_generic.GetAtoms():
        if atom.IsInRing() and not atom.GetIsAromatic():
            atom.SetAtomicNum(6)
    return mol_generic


def calc_murcko_similarity(murcko_query_smi: str, murcko_db_smi: str) -> float:
    try:
        mol1 = Chem.MolFromSmiles(murcko_query_smi)
        mol2 = Chem.MolFromSmiles(murcko_db_smi)
        if mol1 is None or mol2 is None:
            return 0.0

        mol1_gen = generalize_aliphatic_atoms(mol1)
        mol2_gen = generalize_aliphatic_atoms(mol2)

        mcs = rdFMCS.FindMCS(
            [mol1_gen, mol2_gen],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            bondCompare=rdFMCS.BondCompare.CompareOrder
        )
        mcs_atoms = mcs.numAtoms
        query_atoms = mol1.GetNumAtoms()
        return round(mcs_atoms / query_atoms, 3) if query_atoms != 0 else 0.0
    except Exception:
        return 0.0
# ==========================================================


def search_single_skeleton_parallel(df: pd.DataFrame, query_smiles: str, min_sim=0.8, max_workers=16) -> pd.DataFrame:
    query_murcko = get_murcko_smiles(query_smiles)
    if query_murcko is None:
        return pd.DataFrame()
    valid_df = df.dropna(subset=["murcko_scaffold"])
    rows = [row for _, row in valid_df.iterrows()]
    matches = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(calc_murcko_similarity, query_murcko, row["murcko_scaffold"]): row for row in rows}
        for future in tqdm(as_completed(future_to_row), total=len(rows), desc="molecular matching"):
            row = future_to_row[future]
            sim = future.result()
            if sim >= min_sim:
                r = row.copy()
                r["murcko_similarity"] = sim
                matches.append(r)
    return pd.DataFrame(matches)


def batch_search_skeletons_parallel(df: pd.DataFrame, skeletons: list, min_sim=0.8, max_workers=16) -> list:
    results = []
    for idx, smi in enumerate(tqdm(skeletons, desc="total progress", colour="blue")):
        res_df = search_single_skeleton_parallel(df, smi, min_sim, max_workers)
        results.append(res_df)
    return results


def calculate_scaffold_score(df_matched: pd.DataFrame) -> dict:
    if df_matched.empty:
        return {"hit_num": 0, "sum_target": 0, "target_ratio": 0, "mean_target_density": 0, "mean_marketed": 0, "good_num": 0, "bad_num": 0, "good_ratio": 0.0, "final_score": 0.0}

    hit_num = len(df_matched)
    sum_target = df_matched["target_count"].sum()
    good_num = (df_matched["scaffold_label"] == 1).sum()
    bad_num = (df_matched["scaffold_label"] == 0).sum()
    good_ratio = good_num / hit_num if hit_num > 0 else 0
    target_ratio = hit_num / sum_target if sum_target > 0 else 0
    mean_target_density = df_matched["target_density"].mean() if hit_num > 0 else 0
    mean_marketed = df_matched["is_marketed"].mean() if hit_num > 0 else 0

    s1 = target_ratio * 100
    s2 = good_ratio * 100
    S3 = np.clip(mean_target_density * 100, 0, 100)
    S4 = np.clip(mean_marketed * 100, 0, 100)
    S5 = np.clip(hit_num, 0, 100)
    final_score = round((s1 + s2 + S3 + S4 + S5) / 5, 2)

    return {
        "hit_num": hit_num, "sum_target": sum_target, "target_ratio": target_ratio,
        "mean_target_density": mean_target_density, "mean_marketed": mean_marketed,
        "good_num": int(good_num), "bad_num": int(bad_num), "good_ratio": round(good_ratio, 4),
        "final_score": final_score
    }


def main(output_dir: str = '.'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = query_chembl(SQL_QUERY)
    df_clean = clean_data(df_raw)
    df_filtered = filter_pains(df_clean)
    df_aggregated = aggregate_unique_molecules(df_filtered)
    df_labeled = assign_scaffold_label(df_aggregated)

    df_results = batch_search_skeletons_parallel(df_labeled, skeleton_smiles_list, min_sim=0.8, max_workers=16)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    filtered_results = []
    for i, res_df in enumerate(df_results):
        if res_df.empty:
            filtered_results.append(pd.DataFrame())
            continue
        res_df["has_benzo_five_ring"] = res_df["canonical_smiles"].apply(has_benzo_five_membered_ring)
        df_filt = res_df[res_df["has_benzo_five_ring"] == True].copy()
        filtered_results.append(df_filt)

    df_results = filtered_results

    rank_data = []
    for i, d in enumerate(df_results):
        res = calculate_scaffold_score(d)
        res['Scaffold_name'] = f'Scaffold{i+1}'
        rank_data.append(res)

    for i, res_df in enumerate(df_results):
        res_df.to_csv(out / f'df_results{i+1}.csv', index=False, encoding='utf-8-sig')

    df_rank = pd.DataFrame(rank_data).sort_values('final_score', ascending=False).reset_index(drop=True)
    df_rank.to_csv(out / 'final_score.csv', index=False, encoding='utf-8-sig')
    df_labeled.to_csv(out / 'chembl_scaffold_final_labeled.csv', index=False, encoding='utf-8-sig')

    print("Completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChEMBL pipeline')
    parser.add_argument('--out', '-o', default='.', help='OutDir')
    args = parser.parse_args()
    main(args.out)