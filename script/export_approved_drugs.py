from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import sqlite3
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import re

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs


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
where max_phase = 4 and molecule_type = 'Small molecule'
and cp.full_mwt > 200 and cp.full_mwt < 1000
group by cs.canonical_smiles, cs.molregno
"""


chembl_db_path = r"E:\chembl\chembl\36\chembl_36.db"


def standardize(smiles):

    mol = Chem.MolFromSmiles(smiles)

    clean_mol = rdMolStandardize.Cleanup(mol)

    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

    uncharger = rdMolStandardize.Uncharger() 
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

    te = rdMolStandardize.TautomerEnumerator() 
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)


def smi_to_inchi_nochg(smi):
    mol = Chem.MolFromSmiles(smi)
    inchi = Chem.MolToInchi(mol)
    return re.sub("/p\+[0-9]+", "", inchi)


def main():
    conn = sqlite3.connect(chembl_db_path)

    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )
    print(tables.head(20))

    smiles_df = pd.read_sql_query(SQL_QUERY, conn)

    with BlockLogs():
        smiles_df['std_smiles'] = smiles_df.canonical_smiles.progress_apply(standardize)

    df_ok = smiles_df.dropna(subset=["first_approval"]).copy()
    df_ok.first_approval = df_ok.first_approval.astype(int)
    df_ok_nodupe = df_ok.sort_values("first_approval").drop_duplicates("std_smiles").copy()
    len(df_ok), len(df_ok_nodupe)

    with BlockLogs():
        df_ok_nodupe['inchi'] = df_ok_nodupe.std_smiles.apply(smi_to_inchi_nochg)

    df_final_drug = df_ok_nodupe.sort_values("first_approval").drop_duplicates("inchi").copy()
    len(df_final_drug)

    df_final_drug.to_csv("approved_drugs.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()