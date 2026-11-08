# CMPS355 roadmap, proposal, and supervisor-feedback audit

**Date:** 8 November 2026. **Scope:** the completed IEEE manuscript draft (v1.1), accepted proposal, connected `CMPS355_Research/protocol/` course documents and the verified outputs represented in the manuscript. **Meaning of completed:** supported by saved evidence; not a claim of independent prospective clinical replication.

## Course milestones

| Milestone | Course schedule | Status | Evidence and remaining work |
|---|---|---|---|
| Proposal and literature review | Oct 4, 2026 | Completed previously / approved | `protocol/proposal.pdf`; original literature and research question retained. |
| Complete draft for peer review | Nov 8, 2026 | Complete manuscript artifact | IEEEtran A4, title/abstract, Introduction, Methods, Results, Discussion, Conclusion, references, explanatory tables/figures. Note that submission to course system remains the authors' responsibility. |
| Peer review of another student's paper | Nov 15, 2026 | **Future, not completed** | Course will assign a paper. |
| Revised manuscript and point-by-point rebuttal | Nov 22, 2026 | **Future, not completed** | Must address actual reviewer comments after they are received. |
| Research symposium/presentation | Week 14 | **Future, not completed** | Presentation and Q&A are distinct deliverables. |

## Research stages / workstreams

| Workstream | Status | Qualification |
|---|---|---|
| Cohort acquisition, ID alignment and QC description | Complete for accepted dataset | 689 samples, 9,983 common genes, dataset SHA-256 recorded; seven source D3 QC=EXCLUDE samples retained for comparability, with no exclusion sensitivity run. |
| Four analytical-cohort LOCO partitions | Complete | Four separate held-out tests; no cross-fold test-set use for selection. Cohorts rather than six BioProjects are the test units. |
| LASSO / Elastic Net / Stability Selection / StableMate | Executed and verified | Four folds for each. Native fitting conventions and common-learner panels explicitly distinguished. StableMate native fitted R objects were not independently rerun. |
| TENTACLES official native V2 consensus / external scoring | Executed and exported, independently checked at output level | Four recovered/passed native runs; CD-positive orientation, nine-voter six-vote primary and five-vote sensitivity; exported panel votes/held-out predictions checked; original complete RDS not independently refitted. D4 excluded XGBoost had incomplete selected CV resamples. |
| Fixed downstream prediction benchmark | Complete for five methods | Identical class-balanced L2 logistic, C=1, same 4 splits and common feature transformation. Frozen panels retain selector-specific upstream differences. |
| LOCO panel agreement | Complete for five methods | Mean of six pairwise Jaccards plus panel sizes/intersections; six comparisons share development cohorts, therefore not independent gene replication. |
| Independent cohort gene signature comparisons | **Partial across methods** | Independently fitted, patient-disjoint sparse LASSO/Elastic Net signatures quantified; Stability Selection generated within-cohort signatures. A coherent five-method independent discovery comparison was **not** performed. |
| Panel-size sensitivity | **Partial** | Top-four comparisons for LASSO, Elastic Net, Stability Selection, StableMate; top-ten for LASSO, Elastic Net, StableMate. No principled tie-resolved TENTACLES top-four ranking; no five-method size-matched predictive refit. |
| Pathway reproducibility | Complete for five LOCO method panels | Reactome v98, eligible training backgrounds, one-sided over-representation, BH q<0.05; V2 replaced historical V1 pathway values. Pathway comparison remains size- and annotation-sensitive. |
| Uncertainty | Descriptive evidence sufficient for draft | Four-cohort means and between-cohort variability; patient-level stratified bootstrap intervals for verified endpoints. They do not estimate performance across unseen source populations. |
| Training-size subsampling / additional diseases | Optional, not performed | Explicitly made secondary by supervisor; not necessary for November 8 core benchmark. |
| D3 source quality-flag removal sensitivity | Not performed | Targeted robustness follow-up recommended if a reviewer requests or timeline permits; requires controlled refitting and recalculation, not simply dropping rows from score tables. |
| Independent R-native refitting, public turnkey release | Partly verified / not complete | Provenance, saved native outputs and aggregate code available; complete independent refitting of every native model, resolution of all candidate warnings, and a fully public source/data release are not claimed. |

## Supervisor's five comments: precise status

1. **Scope:** satisfied. Four Crohn's analytical cohorts, no additional diseases.
2. **Common predictor:** satisfied for panel evaluation. Same L2 logistic learner and preprocessing; **upstream selection, tuning, and signature sizes are still unmatched**, so this is not causal isolation of the selector.
3. **Gene-count fairness:** **partially addressed**, not fully eliminated. All method panel counts reported, four-method top-four sensitivity plus three-method top-ten analysis. No scientifically justified five-method matched-size analysis due to TENTACLES vote ties and small Stability Selection signatures. Do not describe complete fairness across five methods.
4. **LOCO reproducibility:** satisfied for operational definition and calculation (six pairwise mean Jaccards across four LOCO panels); these are **selection consistency**, not independent patient-cohort replication. Genuinely disjoint cohort signatures are covered only for a subset of methods.
5. **Training-size subsampling:** appropriately deprioritised. Not a requirement for this complete manuscript draft.

## Original proposal questions: strength of evidence

- **Q1 (prediction vs gene consistency):** answered descriptively using common-classifier AUROC and LOCO Jaccards for all five methods. Does not prove a universal or causal trade-off.
- **Q2 (which are most reproducible independently):** only partly answered. Five-way LOCO consistency can be ranked descriptively; independent cohorts have sparse-model fits, but not five complete independent selector discoveries. Avoid wording that implies all five were independently replicated.
- **Q3 (pathway vs gene recurrence):** explored for all five LOCO panel types, with enrichment/size limitations and undefined Jaccards for two methods. Results do not show pathways are always more reproducible.

## What must still run?

**For November 8 draft:** no additional computational experiment is necessary to present the core evidence honestly, provided the submitted manuscript retains its explicit limitations and meets course submission instructions.

**For stronger research / later revision, priority order:**
1. D3 QC-flag exclusion sensitivity across the principal comparison (costly refits; changes training and test composition).
2. Better size-aware sensitivity including TENTACLES using a prespecified vote-tie strategy or method-independent outcome appropriate to incomparable ranks. Do not impose an arbitrary CSV-row ordering.
3. Multi-method, genuinely independent cohort gene-discovery validation (substantial extra native runs; not necessary for the first manuscript draft).
4. Broader source-cohort validation and native model re-audits when feasible. The 124-entry historical registry is an experiment inventory, **not** a requirement to execute 124 runs.

## Manuscript AI disclosure

The added acknowledgment states the specific kinds of assistance without falsely limiting AI use to proofreading or claiming that the research text was generated without assistance. The authors retain responsibility for study claims. If the university or course supplies a more specific AI-use disclosure policy, its exact requirements supersede this concise acknowledgment.

## Source material

- Connected Drive `CMPS355_Research/protocol/{project_guidelines,proposal,from_proposal_to_paper,peer_review_and_rebuttals}.pdf`.
- Connected Drive `CMPS355_Research/docs/study_plan.md` (dated Nov 6; forward-looking stage statuses are superseded by subsequently verified runs).
- Existing manuscript `verification/FINAL_SCIENTIFIC_AUDIT.md` and aggregate results/provenance under `verification/`.
- The original professor's five comments supplied by the authors.
