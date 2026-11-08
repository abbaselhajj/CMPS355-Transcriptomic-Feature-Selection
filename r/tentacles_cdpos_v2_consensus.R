#!/usr/bin/env Rscript
# Official author getConsensus, primary paper-aligned n.min=6 of 9; n.min=5 secondary.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript tentacles_cdpos_v2_consensus.R FIT_RDS NEW_OUTPUT_DIRECTORY')
rds <- normalizePath(args[[1]],mustWork=TRUE)
out <- normalizePath(args[[2]],mustWork=TRUE)
lib <- Sys.getenv('CMPS355_R_LIB','')
if(nzchar(lib) && dir.exists(lib)) .libPaths(c(lib,.libPaths()))
stopifnot(requireNamespace('TENTACLES',quietly=TRUE))
sha <- tolower(utils::packageDescription('TENTACLES')[['RemoteSha']])
if(is.null(sha) || is.na(sha) || sha!='069a26271e3885e015df25baa40a7a1140827d5f') stop('Pinned TENTACLES revision mismatch')
if(file.exists(file.path(out,'consensus_native_finished.rds')) || file.exists(file.path(out,'consensus_nmin5_finished.rds'))) stop('Refuse to overwrite existing native consensus')
fit <- readRDS(rds)
if(!isS4(fit) || !inherits(fit,'runClassifiers.obj')) stop('Not a native official runClassifiers object')
features <- methods::slot(fit,'model.features')
if(!is.list(features) || length(features)!=10L || is.null(names(features)) || anyDuplicated(names(features))) stop('Expected ten named fitted native model feature sets')
models <- names(features)
writeLines(models,file.path(out,'classifier_model_names.txt'))
exclude <- grep('^xgboost($|_)',models,value=TRUE,ignore.case=TRUE)
if(length(exclude)!=1L) stop('Must find exactly one XGBoost to exclude, found: ',paste(exclude,collapse=', '))
writeLines(c('ORIGINAL_TENTACLES_COMMIT=069a26271e3885e015df25baa40a7a1140827d5f',
             'HELDOUT_LABEL=CD','NATIVE_TUNING_METRIC=f_meas','EVENT=CD_FIRST',
             'PRIMARY_CONSENSUS_N_MIN=6','SENSITIVITY_N_MIN=5',
             paste0('EXCLUDED_FROM_9_VOTERS=',exclude),
             'AUTHOR_GETCONSENSUS_TESTS=INTERNAL_DEVELOPMENT_ONLY',
             'OUTER_HELDOUT_COHORT_NOT_USED_FOR_CONSENSUS'),file.path(out,'consensus_protocol.txt'))
extract <- function(result,nmin,tag) {
  if(!is.list(result) || !all(c('consensusGenes','dataFrame') %in% names(result))) stop('Unexpected native getConsensus structure')
  g <- result$consensusGenes
  if(!is.character(g) || anyNA(g) || anyDuplicated(g) || any(!nzchar(g))) stop('Bad consensus gene vector')
  votes <- result$dataFrame
  if(!is.data.frame(votes) && !is.matrix(votes)) stop('Unexpected native vote matrix')
  if(nrow(votes)!=9L) stop('Expected votes from nine remaining classifiers; got ',nrow(votes))
  vs <- as.matrix(votes)
  if(anyNA(vs) || !all(vs %in% c(0,1))) stop('Native votes must be binary and complete')
  cnt <- colSums(vs)
  derived <- names(cnt)[cnt>=nmin]
  if(!setequal(g,derived) || length(g)!=length(derived)) stop('Native consensus disagrees with independently counted votes at threshold ',nmin)
  utils::write.csv(data.frame(gene=g),file.path(out,paste0('consensus_genes_',tag,'.csv')),row.names=FALSE)
  utils::write.csv(data.frame(gene=names(cnt),votes=as.integer(cnt)),file.path(out,paste0('native_votes_',tag,'.csv')),row.names=FALSE)
  cat('CONSENSUS_',tag,'_GENES=',length(g),'\n',sep='')
  invisible(g)
}
set.seed(123L)
cat('AUTHOR_GETCONSENSUS_NMIN6_START\n')
c6 <- TENTACLES::getConsensus(fit,n.min=6L,exclude=exclude,plot=FALSE)
if(is.null(c6)) stop('Author consensus n=6 returned NULL')
g6 <- extract(c6,6L,'nmin6')
tmp <- file.path(out,'consensus_native_finished.rds.partial')
saveRDS(c6,tmp,compress='gzip')
if(!file.rename(tmp,file.path(out,'consensus_native_finished.rds'))) stop('Could not finalize nmin6 RDS')
utils::write.csv(data.frame(gene=g6),file.path(out,'consensus_genes_native.csv'),row.names=FALSE)
rm(c6);gc(verbose=FALSE)
set.seed(123L)
cat('AUTHOR_GETCONSENSUS_NMIN5_SENSITIVITY_START\n')
c5 <- TENTACLES::getConsensus(fit,n.min=5L,exclude=exclude,plot=FALSE)
if(is.null(c5)) stop('Author consensus n=5 returned NULL')
g5 <- extract(c5,5L,'nmin5')
if(!all(g6 %in% g5)) stop('Six-vote panel must be a subset of five-vote panel')
tmp <- file.path(out,'consensus_nmin5_finished.rds.partial')
saveRDS(c5,tmp,compress='gzip')
if(!file.rename(tmp,file.path(out,'consensus_nmin5_finished.rds'))) stop('Could not finalize nmin5 RDS')
writeLines(capture.output(str(c5,max.level=2)),file.path(out,'consensus_native_structure.txt'))
cat('OFFICIAL_CONSENSUS_6_AND_5_PASS=TRUE\n')
