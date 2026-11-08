# Stage 9 / preflight: read-only R-package and verified D1-preprocess inspection.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop("Usage: Rscript tentacles_stage9_preflight.R preprocessed.rds report_directory expected_source_sha")
input <- normalizePath(args[[1]], mustWork=TRUE)
out <- normalizePath(args[[2]], mustWork=TRUE)
expected <- tolower(args[[3]])
lib <- Sys.getenv("CMPS355_R_LIB", "")
if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib, .libPaths()))
if (!requireNamespace("TENTACLES", quietly=TRUE)) stop("Official TENTACLES package is not installed")
meta <- utils::packageDescription("TENTACLES")
sha <- tolower(meta[["RemoteSha"]])
if (length(sha) != 1L || is.na(sha) || sha != expected) stop("Official package source SHA mismatch")
if (as.character(utils::packageVersion("TENTACLES")) != "0.1.0") stop("Unexpected TENTACLES package version")
x <- readRDS(input)
if (!inherits(x, "preProcess.obj")) stop("Input is not an official preProcess.obj")
functions <- c("preProcess", "runClassifiers", "getConsensus", "testConsensus", "valConsensus")
if (!all(functions %in% getNamespaceExports("TENTACLES"))) stop("Missing a published TENTACLES function")
required <- c("tidymodels", "workflowsets", "recipes", "themis", "colino", "Boruta", "ranger", "nnet", "kernlab", "C50", "bonsai", "lightgbm", "rpart", "earth", "xgboost", "mixOmics", "plsmod", "yardstick")
loaded <- vapply(required, requireNamespace, logical(1), quietly=TRUE)
report <- data.frame(package=required, installed=unname(loaded), stringsAsFactors=FALSE)
utils::write.csv(report, file.path(out,"optional_dependency_check.csv"), row.names=FALSE)
writeLines(capture.output(utils::sessionInfo()), file.path(out,"session_info.txt"))
writeLines(capture.output(str(x, max.level=2)), file.path(out,"preprocessed_structure.txt"))
for (f in functions) {
  fn <- getExportedValue("TENTACLES", f)
  writeLines(capture.output(print(formals(fn))), file.path(out, paste0("formals_", f, ".txt")))
  tryCatch({
    h <- utils:::.getHelpFile(utils::help(f, package="TENTACLES"))
    writeLines(capture.output(tools::Rd2txt(h)), file.path(out, paste0("help_",f,".txt")))
  }, error=function(e) {
    writeLines(conditionMessage(e), file.path(out, paste0("help_",f,"_unavailable.txt")))
  })
}
cat("TENTACLES_SHA=",sha,"\n",sep="")
cat("TENTACLES_VERSION=",as.character(utils::packageVersion("TENTACLES")),"\n",sep="")
cat("R_VERSION=",as.character(getRversion()),"\n",sep="")
cat("D1_PREPROCESSED_CLASS=",paste(class(x),collapse=","),"\n",sep="")
cat("OPTIONAL_DEPENDENCIES_MISSING=",paste(required[!loaded],collapse=","),"\n",sep="")
if (any(!loaded)) stop("Preflight stopped: restore/install missing optional packages before a 10-model run; see optional_dependency_check.csv")
# Tiny synthetic smoke: test actual authors' classifier API, F1 tuning, and downsampling
# without using any held-out cohort outcome or the publication's D1 labels.
set.seed(1901L)
ns <- 48L; ng <- 60L
syn <- matrix(stats::rpois(ns*ng,lambda=40),nrow=ns,ncol=ng)
rownames(syn) <- sprintf("SYN_%03d",seq_len(ns))
colnames(syn) <- sprintf("SYNGENE%03d",seq_len(ng))
syn[25:48,1:10] <- syn[25:48,1:10] + 40
synthetic_clin <- data.frame(class=c(rep("Not IBD",24L),rep("CD",24L)),row.names=rownames(syn))
synthetic_preprocessed <- TENTACLES::preProcess(as.data.frame(syn),synthetic_clin,
 class="class",case.label="CD", mincpm=1,minfraction=0.1,
 data.type="rnaseq",is.normalized=FALSE,batch=NULL,covar.mod=NULL,plot=FALSE)
if (!inherits(synthetic_preprocessed,"preProcess.obj")) stop("Synthetic official preProcess did not return expected class")
set.seed(1901L)
synthetic_fit <- TENTACLES::runClassifiers(preProcess.obj=synthetic_preprocessed,
 models="decision_tree",selector.recipes="base",tuning.method="tune_grid",
 n=2,v=2,boruta.maxRuns=100,selector.threshold=0.95,
 metric="f_meas",nsim=2,filter=FALSE,seed=1901,
 downsample=TRUE,plot=FALSE,parallel=FALSE)
if (!inherits(synthetic_fit,"runClassifiers.obj")) stop("Official synthetic runClassifiers did not return expected class")
saveRDS(synthetic_fit,file.path(out,"synthetic_classifier_smoke.rds"))
writeLines(capture.output(str(synthetic_fit,max.level=2)),file.path(out,"synthetic_classifier_structure.txt"))
cat("STAGE9_SYNTHETIC_CLASSIFIER_PASS\n")
cat("STAGE9_PREFLIGHT_PASS\n")
