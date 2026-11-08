# Official authors' classifier implementation, applying the published D1 discovery recipe.
# Paper-comparison track only. NOT our four-way leave-one-cohort-out test.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop("Usage: Rscript tentacles_stage9_paper_d1_classifiers.R d1_preprocessed.rds output_dir expected_source_sha")
input <- normalizePath(args[[1]], mustWork=TRUE)
out <- normalizePath(args[[2]], mustWork=TRUE)
expected <- tolower(args[[3]])
lib <- Sys.getenv("CMPS355_R_LIB", "")
if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib, .libPaths()))
if (!requireNamespace("TENTACLES", quietly=TRUE)) stop("TENTACLES package not found")
installed <- tolower(utils::packageDescription("TENTACLES")[["RemoteSha"]])
if (!identical(installed,expected) || as.character(utils::packageVersion("TENTACLES"))!="0.1.0") stop("Package pin mismatch")
if (file.exists(file.path(out,"runclassifiers_finished.rds"))) stop("Refusing to overwrite a completed classifiers RDS")
prep <- readRDS(input)
if (!inherits(prep,"preProcess.obj")) stop("Input is not the authors' preprocessing object")
set.seed(123L)
Sys.setenv(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
options(warn=1)
models <- c("rand_forest", "mlp", "bag_mlp", "svm_linear", "C5_rules", "lightGBM", "decision_tree", "mars", "xgboost", "pls")
selectors <- c("corr", "roc", "roc", "boruta", "boruta", "roc", "boruta", "boruta", "base", "boruta")
stopifnot(length(models)==10L, length(selectors)==10L)
config <- data.frame(model=models,selector=selectors,stringsAsFactors=FALSE)
utils::write.csv(config,file.path(out,"paper_model_selector_pairs.csv"),row.names=FALSE)
writeLines(c(
 "TRACK=published_D1_discovery_adapted_to_aligned_source",
 "NOT_LOCO=TRUE",
 paste0("SOURCE_SHA=",expected),
 "MODEL_COUNT=10", "TUNING=tune_grid", "N_CANDIDATES=5", "CV_FOLDS=5",
 "METRIC=f_meas", "DOWN_SAMPLE=TRUE", "SELECTOR_THRESHOLD=0.95",
 "BORUTA_MAXRUNS=100", "VIP_NSIM=2", "GENE_ANNOTATION_FILTER=TRUE",
 "SEED=123", "PLOT=FALSE", "PARALLEL=FALSE",
 "NOTE=Model pairs and F1 tuning from publication; exact authors' grid draws/randomness not certified"
), file.path(out,"protocol.txt"))
cat("CLASSIFIERS_START=TRUE\n", "TRACK=PAPER_D1_DISCOVERY\n", "SOURCE_SHA=",expected,"\n",sep="")
start <- proc.time()[[3]]
fit <- TENTACLES::runClassifiers(
  preProcess.obj=prep,
  models=models,
  selector.recipes=selectors,
  tuning.method="tune_grid", n=5, v=5,
  boruta.maxRuns=100, selector.threshold=0.95,
  metric="f_meas", nsim=2, filter=TRUE,
  seed=123, downsample=TRUE, plot=FALSE, parallel=FALSE
)
elapsed <- proc.time()[[3]] - start
if (is.null(fit)) stop("runClassifiers returned NULL")
# Save full native output first: evidence even if later schema checks fail.
partial <- file.path(out,"runclassifiers_finished.rds.partial")
saveRDS(fit, partial, compress="gzip")
if (!file.rename(partial,file.path(out,"runclassifiers_finished.rds"))) stop("Could not finalize native RDS")
writeLines(capture.output(str(fit,max.level=2)), file.path(out,"runclassifiers_structure.txt"))
if (!inherits(fit,"runClassifiers.obj")) stop("Unexpected native result class; model RDS preserved for inspection")
if (!is.list(fit[["model.features"]]) || length(fit[["model.features"]])!=10L) stop("Native result did not include all 10 model feature-importance entries")
writeLines(capture.output(utils::sessionInfo()), file.path(out,"r_session_info.txt"))
writeLines(as.character(elapsed), file.path(out,"elapsed_seconds.txt"))
if (is.list(fit)) {
  if (!is.null(fit[["model.features"]])) {
    saveRDS(fit[["model.features"]], file.path(out,"model_features_native.rds"))
    writeLines(capture.output(str(fit[["model.features"]],max.level=3)), file.path(out,"model_features_structure.txt"))
  }
  if (!is.null(fit[["performances"]])) {
    saveRDS(fit[["performances"]], file.path(out,"performances_native.rds"))
    writeLines(capture.output(str(fit[["performances"]],max.level=3)), file.path(out,"performances_structure.txt"))
  }
}
cat("CLASSIFIERS_RDS_SAVED=TRUE\n",sep="")
cat("CLASSIFIERS_ELAPSED_SECONDS=",sprintf("%.2f",elapsed),"\n",sep="")
cat("CLASSIFIERS_OBJECT_CLASS=",paste(class(fit),collapse=","),"\n",sep="")
cat("NEXT_GATE=inspect_native_importance_and_getConsensus_API\n")
cat("CLASSIFIERS_PASS=TRUE\n")
