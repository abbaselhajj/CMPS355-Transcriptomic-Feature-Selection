# CMPS355 project: proposal-to-evidence research gap audit

**Sources reviewed:** approved `proposal.pdf`, `project_guidelines.pdf`, `from_proposal_to_paper.pdf`, existing Drive runs and their saved JSON/CSV reports as of 2026-11-08. This is a gap audit, not a claim that any new analysis not explicitly listed here has run.

## Research question / scope
> How do transcriptomic feature-selection strategies differ in the trade-off between predictive performance and biomarker reproducibility across independent cohorts?

**Q1:** Does greater gene-level reproducibility necessarily reduce predictive performance?  
**Q2:** Which strategies give the most reproducible signatures under independent-cohort evaluation?  
**Q3:** Is pathway-level reproducibility consistently higher than gene-level, and does this differ by method?

## Verified completed project components
- Dataset-source reconstruction: recovered original sample IDs from `.RData`, independently verified all 689 count profiles and metadata against accepted corrected ZIP, and recorded original CSV row misalignment. Pipeline Stage 0A execution exists in Drive.
- Stage 1A preprocessing and data exploration; Stage 12 cohort characterization (PCA by cohort/BioProject, class balance, library-size QC) and train-only CPM filtering are saved in Drive.
- Four LOCO outer test folds of LASSO, elastic net, stability selection, StableMate and official TENTACLES with private predictions and appropriate provenance; all four original TENTACLES consensus panels independently checked.
- Existing Stage 22 common-classifier study across five selectors with a fixed logistic model, four heldout cohorts, numerical audit and saved figures.
- Independent-cohort sparse LASSO/EN signatures; saved `runs/independent_signatures/d1..d4` including gene coefficients/subsampling and metrics.
- Reactome pathway analysis for **LASSO/Elastic Net** signatures, source `runs/pathway_analysis/20261107T103148Z/` and independent-cohort pathway figures; Stage16 top-K pathway-size sensitivity (`top_k=25,50,100,200,500`, 40 signatures, 60 pairwise comparisons), report PASS.
- TENTACLES original n.min=5 and exploratory n.min=6 sensitivity; native classifier scores inspected with valid numerical CV results but old Not-IBD-positive orientation. This V2 patch only prepares corrected native rerunning; **V2 training and new external predictions have not yet been executed on WSL**.

## PRIORITY 1 — before claiming a fully consistent method comparison
1. **Complete V2 native CD-positive TENTACLES**, retain original five-vote outcomes, publish primary six-vote comparison separately. The corrected new external performance is unknown until V2 folds actually run.
2. **Refresh the common downstream-model comparison** on **V2 six-vote gene panels** (same fixed logistic and train-only transform as other methods), with method identifiers and provenance. Old Stage22 TENTACLES panel represents V1; never substitute it for corrected V2. Do not claim the existing baseline ranking survives the rerun.
3. **Unify per-method AUROC/balanced accuracy/F1, signature sizes, pairwise gene Jaccard/selection frequency into a single prediction–reproducibility trade-off table/figure**, for exactly the comparison method panels. Explicitly note overlapping development sets make LOCO Jaccard exploratory.
4. **Extend Reactome pathway enrichment / overlap to new native V2 TENTACLES panels**, and to StableMate/stability-selection panels wherever the same background and mapping permit. The completed pathway reports chiefly cover LASSO and elastic net; manuscript must not imply every method underwent pathway reproducibility tests.

## PRIORITY 2 — manuscript-level interpretive work
5. Include method-sensitive uncertainty, limited number of cohorts (n=4 independent heldout units), target imbalance and cohort/study composition; report which estimates are descriptive rather than statistically independent. Explain which models have native vs shared logistic downstream predictors and do **not** compare their native CV F1 with heldout F1.
6. Evaluate **rank correlation where meaningful and available** (native importance rankings vs sparse coefficients need careful differing support/scale); if not comparable, justify omission explicitly rather than forcing arbitrary rankings.
7. Consider a complementary size-matched panel comparison, e.g., common top-K where methods provide defensible ranked genes; report as a planned sensitivity analysis, not a posthoc tuning using test results. Existing pathway top-K results do NOT constitute method-wide prediction vs training-size subsampling.
8. The proposal offers **controlled training-sample-size subsampling only where computationally feasible**. It has not been implemented across all methods; document as optional future analysis, not a mandatory completed result.
9. Source data originate from published TENTACLES Crohn's cohorts; they are independent in the **outer test fold** but not entirely new to the author paper. Avoid overclaim of external biological or clinical validation.
10. Include methods/provenance reproducibility, exact positive class for every metric, comparison predeclared vs exploratory analyses, FDR-adjusted pathway analyses and caveats about feature selection + upstream tuning differences.

## Manuscript deadline and structure
The official CMPS355 guidelines request a **complete manuscript draft** due Nov 8 (title/abstract, introduction, methods, results, discussion, figures/tables, references). The user also requests 8–12 pages including references; report only finished experiments with results, qualify uncompleted reruns and prioritize submitting a defensible complete draft instead of overpromising extra training. Peer review and rebuttal follow in later milestones. The provided `from_proposal_to_paper` teaching notes emphasize a reasoned argument from evidence, not an experiment diary.

## Non-issues to avoid redoing
- Canonical data alignment has already passed independent value-level verification; do not rebuild it again for this rerun.
- The original 5-vote TENTACLES Stage10 heldout CD-positive metrics are mathematically valid as V1 workflow results even if native tuning targeted controls.
- Source RDS and model/log history are scientifically valuable; do not delete for a clean public GitHub checkout. Release only curated aggregates; keep patient-level outputs private.
