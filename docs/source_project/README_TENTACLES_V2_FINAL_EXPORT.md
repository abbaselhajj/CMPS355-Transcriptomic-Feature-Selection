# Final TENTACLES V2 results export — read-only

This helper exports the COMPLETE four-fold V2 primary six-vote and secondary five-vote held-out results, official native selected genes, model votes, small provenance text and D3/D4 recovery evidence into a single analysis ZIP. No native RDS or binary classifier files are copied. No new model is fitted.

From your existing project root:

```bash
unzip -n CMPS355_TENTACLES_V2_Final_Export_Helper.zip
bash scripts/export_tentacles_v2_final.sh
```

Attach the resulting `CMPS355_TENTACLES_V2_Final_Results_For_Analysis.zip` to the chat. It includes patient-level held-out prediction rows and must be treated as **private**. The assistant can then carry out the final five-selector, matched downstream-predictor analysis and reproducibility/Reactome updates with the already saved four other methods in Drive.

The export refuses if a fold is not COMPLETE, a native panel does not agree with its nine-voter threshold, Stage10 markers are missing, patient predictions are incomplete/duplicated, or external metrics disagree with the saved macro summary.

This helper does not claim that all ten native classifiers had five valid CV resamples. The D4 XGBoost exception is preserved in the recovery evidence and excluded from the predefined nine-voter consensus.
