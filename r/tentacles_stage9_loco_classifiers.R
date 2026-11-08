# The official TENTACLES selectors/classifiers applied to the three discovery
# cohorts of a four-way LOCO split. No test samples ever enter this R process.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=5L) stop("Usage: Rscript ... train_counts.csv train_clinical.csv output_dir holdout pinned_git_sha")
csv_counts <- normalizePath(args[[1]],mustWork=TRUE)
csv_clinical <- normalizePath(args[[2]],mustWork=TRUE)
out <- normalizePath(args[[3]],mustWork=TRUE)
heldout <- args[[4]]
expected <- tolower(args[[5]])
if (!heldout %in% c("d1","d2","d3","d4")) stop("Invalid heldout label")
lib <- Sys.getenv("CMPS355_R_LIB", "")
if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib,.libPaths()))
if (!requireNamespace("TENTACLES",quietly=TRUE)) stop("Official TENTACLES unavailable")
if (tolower(utils::packageDescription("TENTACLES")[["RemoteSha"]]) != expected ||
    as.character(utils::packageVersion("TENTACLES"))!="0.1.0") stop("Package source/version mismatch")
counts <- utils::read.csv(csv_counts,check.names=FALSE,stringsAsFactors=FALSE)
clin <- utils::read.csv(csv_clinical,check.names=FALSE,stringsAsFactors=FALSE)
if (!all(c("ID","Study","class","cohort") %in% names(clin)) || !"ID" %in% names(counts)) stop("Training input missing metadata")
if (!identical(as.character(counts$ID),as.character(clin$ID))) stop("Training sample alignment failed")
if (anyDuplicated(counts$ID)||anyNA(counts$ID)||anyDuplicated(names(counts))) stop("Duplicate IDs/features")
if (heldout %in% unique(clin$cohort) || length(unique(clin$cohort)) != 3L) stop("Outer test cohort seen in training!")
if (!identical(sort(unique(clin$class)),c("CD","Not IBD"))) stop("Unexpected classes")
if (nrow(counts)!= 689L - switch(heldout,d1=260L,d2=179L,d3=76L,d4=174L)) stop("Wrong number of training patients")
row.names(counts) <- counts$ID
counts$ID <- NULL
row.names(clin) <- clin$ID
clin <- data.frame(class=clin$class,row.names=row.names(clin),stringsAsFactors=FALSE)
if (ncol(counts)!=9983L || !all(vapply(counts,is.numeric,logical(1)))) stop("Unexpected harmonized feature matrix")
if (anyNA(counts) || any(!is.finite(as.matrix(counts))) || any(as.matrix(counts)<0)) stop("Invalid count data")
if (any(rowSums(as.matrix(counts))<=0)) stop("Zero library-size sample")
if (file.exists(file.path(out,"runclassifiers_finished.rds"))) stop("Will not overwrite a native completed model")
Sys.setenv(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
options(warn=1)
set.seed(123L)
cat("LOCO_TRAIN_ONLY_PASS=TRUE\n","HELDOUT=",heldout,"\n","TRAIN_PATIENTS=",nrow(counts),"\n",sep="")
start <- proc.time()[[3L]]
prep <- TENTACLES::preProcess(counts,clin,class="class",case.label="CD",
  mincpm=1,minfraction=0.1,data.type="rnaseq",is.normalized=FALSE,
  batch=NULL,covar.mod=NULL,plot=FALSE)
if (!inherits(prep,"preProcess.obj")) stop("Official LOCO train-only preprocessing failed")
saveRDS(prep,file.path(out,"train_only_preprocessed.rds"),compress="gzip")
models <- c("rand_forest","mlp","bag_mlp","svm_linear","C5_rules","lightGBM","decision_tree","mars","xgboost","pls")
selectors <- c("corr","roc","roc","boruta","boruta","roc","boruta","boruta","base","boruta")
utils::write.csv(data.frame(model=models,selector=selectors),file.path(out,"model_selector_pairs.csv"),row.names=FALSE)
writeLines(c(paste0("HELDOUT=",heldout),"TRACK=official_TENTACLES_LOCO_train_only",
  "SCALING_AND_SELECTION=author_preProcess_and_recipes_train_only",
  "CV=5_fold_within_discovery_set_not_inner_LOCO", "TUNING=tune_grid;5_candidates;metric=f_meas",
  "DOWNSAMPLE=TRUE", "BATCH_CORRECTION=NONE", "GO_KEGG_ANNOTATION_FILTER=TRUE", "CONSENSUS_AND_TEST_PREDICTIONS=not_yet_computed",
  "VALIDATION=external_heldout_cohort_never_seen_by_R_fit"),file.path(out,"protocol.txt"))
cat("OFFICIAL_RUNCLASSIFIERS_START=TRUE\n")
library(TENTACLES)
fit <- TENTACLES::runClassifiers(preProcess.obj=prep, models=models,
   selector.recipes=selectors,tuning.method="tune_grid",n=5,v=5,
   boruta.maxRuns=100,selector.threshold=0.95,metric="f_meas",
   nsim=2,filter=TRUE,seed=123,downsample=TRUE,plot=FALSE,parallel=FALSE)
if (is.null(fit)) stop("Official author classification returned NULL")
partial <- file.path(out,"runclassifiers_finished.rds.partial")
saveRDS(fit,partial,compress="gzip")
if (!file.rename(partial,file.path(out,"runclassifiers_finished.rds"))) stop("Failed atomic RDS finalization")
writeLines(capture.output(str(fit,max.level=2)),file.path(out,"runclassifiers_structure.txt"))
if (!inherits(fit,"runClassifiers.obj")) stop("Unexpected native result class; model RDS preserved")
if (!is.list(methods::slot(fit,"model.features")) || length(methods::slot(fit,"model.features"))!=10L) stop("Missing any of the 10 native model feature lists")
writeLines(capture.output(utils::sessionInfo()),file.path(out,"r_session_info.txt"))
if (isS4(fit) && "model.features" %in% methods::slotNames(fit)) saveRDS(methods::slot(fit,"model.features"),file.path(out,"model_features_native.rds"))
if (isS4(fit) && "performances" %in% methods::slotNames(fit)) saveRDS(methods::slot(fit,"performances"),file.path(out,"performances_native.rds"))
cat("CLASSIFIERS_ELAPSED_SECONDS=",sprintf("%.2f",proc.time()[[3L]]-start),"\n",sep="")
cat("CLASSIFIERS_PASS=TRUE\nNEXT_GATE=author_getConsensus_and_external_training_only_classifier\n")
