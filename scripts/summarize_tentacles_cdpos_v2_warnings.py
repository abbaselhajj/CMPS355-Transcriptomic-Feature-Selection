#!/usr/bin/env python3
"""Summarize historical native warning patterns. Zero warnings is NOT assumed."""
import json,re,sys
from pathlib import Path
if len(sys.argv)!=2: raise SystemExit('Usage: python summarize_tentacles_cdpos_v2_warnings.py FOLD_RUN_DIR')
p=Path(sys.argv[1]); t=(p/'classifiers.log').read_text(errors='replace')
items={'mlp_weight_limit':r'too many \([^\n]{0,120}weights|too many weights',
       'lightgbm_invalid_feature_fraction':r'feature_fraction_bynode',
       'boruta_missing_importance':r'getImp result contains NA\(s\) or NaN\(s\)',
       'tune_fit_errors':r'error in (?:tune|fit|\w+\.fit)|\bError:\b|all models failed',
       'warnings_generic':r'(?im)^warning(?::| message:)'}
report={'fold_run':str(p.resolve()),'review_required':True,'warning_counts':{key:len(re.findall(pattern,t,re.I)) for key,pattern in items.items()},
 'interpretation':'A warning count is not a count of failed final models; inspect selected CV metrics and importance quality gates.'}
(p/'warning_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('NATIVE_CANDIDATE_WARNING_COUNTS='+json.dumps(report['warning_counts'],sort_keys=True))
