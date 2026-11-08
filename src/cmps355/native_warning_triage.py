"""Limited classification of TENTACLES log warning receipts (NOT native-fit certification)."""
from __future__ import annotations
import pandas as pd

def report_from_summary(path):
    rows=[]
    x=pd.read_csv(path)
    for _,r in x.iterrows():
        rows.append(dict(heldout=r['heldout'],n_consensus=int(r['native_gene_count']),
          mlp_invalid_weight_tuning_messages=int(r['mlp_weight_errors']),
          lightgbm_invalid_fraction_tuning_messages=int(r['lightgbm_feature_fraction_errors']),
          boruta_missing_importance_warnings=int(r['boruta_nan_warnings']),
          native_RDS_individually_inspected=False,
          interpretation='Risk flag: inspect tuned candidate failures, final fitted importance and vote table before declaring all nine voters reliable'))
    return pd.DataFrame(rows)
