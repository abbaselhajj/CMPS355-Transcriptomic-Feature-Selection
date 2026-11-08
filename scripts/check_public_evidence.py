#!/usr/bin/env python3
"""Recompute the public five-method numeric consistency checks without private data."""
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
base=root/'results/final_five_method'
a=pd.read_csv(base/'five_method_controlled_by_cohort.csv')
b=pd.read_csv(base/'five_method_controlled_macro.csv')
p=pd.read_csv(base/'final_selected_gene_panels.csv')
assert len(a)==20
assert len(p.groupby(['heldout','method']))==20
for _,r in b.iterrows():
 d=a[a.method==r.method]
 assert len(d)==4
 assert abs(d.auroc.mean()-r.mean_auroc)<1e-10
 assert abs(d.balanced_accuracy.mean()-r.mean_balanced_accuracy)<1e-10
print('PASS: 20 method-fold evaluations; all macro means reproduced; 20 selected-gene panels recorded.')
