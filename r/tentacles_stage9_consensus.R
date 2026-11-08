# Official getConsensus() on an already finished ten-model train-only fit.
# This NEVER reads outer held-out data, and never alters the fitted classifier object.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop("Usage: Rscript tentacles_stage9_consensus.R completed_rds output_dir")
f <- normalizePath(args[[1]],mustWork=TRUE)
out <- normalizePath(args[[2]],mustWork=TRUE)
lib <- Sys.getenv("CMPS355_R_LIB",unset="")
if(nzchar(lib)&&dir.exists(lib)) .libPaths(c(lib,.libPaths()))
stopifnot(requireNamespace("TENTACLES",quietly=TRUE))
if(tolower(utils::packageDescription("TENTACLES")[["RemoteSha"]]) != "069a26271e3885e015df25baa40a7a1140827d5f") stop("TENTACLES revision mismatch")
fit <- readRDS(f)
if(!inherits(fit,"runClassifiers.obj")) stop("Input does not contain native runClassifiers.obj")
features <- methods::slot(fit,"model.features")
if(!is.list(features)||length(features)!=10L) stop("Expected ten official classifier feature sets")
observed <- names(features)
writeLines(observed,file.path(out,"classifier_model_names.txt"))
if(is.null(observed)||anyNA(observed)||anyDuplicated(observed)) stop("Cannot safely identify model names")
exclude <- grep("^xgboost($|_)", observed, value=TRUE, ignore.case=TRUE)
if(length(exclude)!=1L) stop("Expected exactly one XGBoost model; inspect classifier_model_names.txt")
if(file.exists(file.path(out,"consensus_native_finished.rds"))) stop("Refusing to overwrite existing official consensus")
cat("CONSENSUS_START=TRUE\n","EXCLUDE_CLASSIFIER=",exclude,"\n","MIN_VOTERS=5\n",sep="")
# Published majority rule among nine contributing models, prespecified excluding XGBoost.
# getConsensus() also invokes testConsensus() on the TRAINING dataset inside fit;
# that internal MLP is NOT a held-out LOCO prediction.
result <- TENTACLES::getConsensus(fit,n.min=5L,exclude=exclude,plot=FALSE)
if(is.null(result)) stop("Official getConsensus() returned NULL")
partial <- file.path(out,"consensus_native_finished.rds.partial")
saveRDS(result,partial,compress="gzip")
if(!file.rename(partial,file.path(out,"consensus_native_finished.rds"))) stop("Cannot atomically save official consensus")
writeLines(capture.output(str(result,max.level=3)),file.path(out,"consensus_native_structure.txt"))
writeLines(capture.output(names(result)),file.path(out,"consensus_native_fields.txt"))
writeLines(c("CONSENSUS_METHOD=authors_getConsensus", "N_MIN=5",
             paste0("EXCLUDE=",exclude),"TRAINING_ONLY=TRUE",
             "NOTE=getConsensus invokes internal training-dataset testConsensus; never use as outer-test prediction"),
           file.path(out,"consensus_protocol.txt"))
# Only export direct gene vector if it is UNAMBIGUOUS in the returned object.
# If not, the native serialized object is preserved for schema-specific extraction.
valid_names <- c("consensus.genes","consensusGenes","consensus_genes","genes","gene.list","gene_list")
if(is.list(result)) {
  matches <- intersect(names(result),valid_names)
  matches <- matches[vapply(result[matches],function(v) is.character(v)&&is.null(dim(v)),logical(1))]
  if(length(matches)==1L && all(nzchar(result[[matches]])) && !anyDuplicated(result[[matches]])) {
    utils::write.csv(data.frame(gene=result[[matches]]),file.path(out,"consensus_genes_native.csv"),row.names=FALSE)
    cat("GENE_VECTOR_FIELD=",matches,"\n","GENE_COUNT=",length(result[[matches]]),"\n",sep="")
  } else cat("GENE_VECTOR_EXPORT=DEFERRED_NATIVE_OBJECT_SAVED\n")
}
cat("OFFICIAL_CONSENSUS_PASS=TRUE\n")
