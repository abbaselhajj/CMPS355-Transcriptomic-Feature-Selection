# CMPS355 IEEE manuscript v1.1

This is an evidence-integrated draft prepared for CMPS 355 graduate peer review. Compile with `bash build.sh` or upload the directory to Overleaf (IEEEtran, A4). The PDF is included.

## Key source provenance
- `protocol/` on connected Drive: official course project guidelines, approved proposal, From Proposal to Paper, Peer Review and Rebuttals.
- Corrected aligned dataset checksum: `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`.
- Private, user-supplied `CMPS355_TENTACLES_V2_Final_Results_For_Analysis(1).zip` independently checked: 148 constituent-file SHA-256 matches, 689 test samples for both consensus thresholds, all selected-gene voting thresholds, metrics and original identifiers. The private archive is **not included in this shareable manuscript ZIP**.
- V2 common logistic classifier was independently refitted in scikit-learn 1.8.0, reproducing frozen probabilities with maximum absolute discrepancy <4e-5; historic four other method panels and metrics kept from audited Stage 22.
- Reactome v98 reference checksum `6056fcd95dc8f09740c8bb64c2765bea1386ce91b9b19cd10bcd45d30451805c`; eligible backgrounds from original Stage 15. Eight LASSO/Elastic Net benchmark enrichment cases exactly reproduce original results; TENTACLES V2 enrichment recomputed, not substituted from V1.
- `verification/` contains only aggregate numerical tables and audit scripts; it deliberately excludes patient-level predictions, identifiers and private data.

## Scope and remaining limitations
Native RDS model objects are absent from the compact export and were not independently retrained. D3/D4 preservation/recovery provenance and unresolved MLP/Boruta/LightGBM warnings remain. A sensitivity analysis excluding seven D3 source-QC-flagged records was not run. Four observational cohorts cannot prove clinical biomarker validation or statistical method superiority.

## Provenance and formatting
Figures are freshly generated from the aggregate data in `verification/`, not stock illustrations. Table and figure captions define each metric, cohort and relevant limitation. Page limit is checked after `pdflatex`/BibTeX.

## Disclosure and scope audit
Version 1.1 adds a concise AI-assistance acknowledgment without changing any experimental results. It also includes `verification/ROADMAP_AND_PROFESSOR_AUDIT.md` distinguishing completed analyses from optional robustness work and later course milestones.

## Project repository

https://github.com/abbaselhajj/CMPS355-Transcriptomic-Feature-Selection
