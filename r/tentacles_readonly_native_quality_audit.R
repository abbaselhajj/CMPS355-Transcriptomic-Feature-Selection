#!/usr/bin/env Rscript
# Read-only and single-fold: inspect an existing native TENTACLES fit/consensus.
# NO fitting, selection, refitting, checkpoint changes or resampling.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript tentacles_readonly_native_quality_audit.R FOLD_COMPLETED_DIR OUTPUT_DIR')
src <- normalizePath(args[[1L]],mustWork=TRUE)
dst <- normalizePath(dirname(args[[2L]]),mustWork=TRUE)
dst <- file.path(dst,basename(args[[2L]]))
if(dir.exists(dst) && length(list.files(dst,all.files=TRUE,no..=TRUE))>0) stop('Refusing to overwrite audit outputs')
dir.create(dst,recursive=TRUE,showWarnings=FALSE)
if(!requireNamespace('TENTACLES',quietly=TRUE)) stop('Pinned TENTACLES package unavailable')
need <- file.path(src,c('runclassifiers_finished.rds','consensus_native_finished.rds','consensus_genes_native.csv','STATUS.txt'))
if(!all(file.exists(need))) stop('Missing native training receipts: ',paste(need[!file.exists(need)],collapse=', '))
if(trimws(readLines(file.path(src,'STATUS.txt'),n=1))!='PASS_NATIVE_CLASSIFIERS_AND_CONSENSUS') stop('Not a complete native fit')
con <- readRDS(file.path(src,'consensus_native_finished.rds'))
if(!is.list(con) || is.null(con$consensusGenes) || is.null(con$dataFrame)) stop('Native consensus format differs; inspect manually')
gene <- as.character(con$consensusGenes)
archived <- utils::read.csv(file.path(src,'consensus_genes_native.csv'),stringsAsFactors=FALSE)
if(!identical(gene,archived$gene)) stop('Consensus gene CSV differs from RDS')
mat <- as.matrix(con$dataFrame)
if(nrow(mat)!=9L || !is.numeric(mat) || any(!is.finite(mat)) || any(mat!=0 & mat!=1)) stop('Consensus model-vote table invalid')
votes <- colSums(mat)
if(length(gene)!=sum(votes>=5L) || !setequal(gene,names(votes)[votes>=5L])) stop('Native majority vote does not match gene panel')
utils::write.csv(data.frame(gene=names(votes),votes=as.integer(votes),in_consensus=names(votes)%in%gene),file.path(dst,'native_vote_integrity.csv'),row.names=FALSE)
rm(con,mat);gc(verbose=FALSE)
cat('Loading trained classifier RDS; ensure there is sufficient RAM.\n',file=stdout())
fit <- readRDS(file.path(src,'runclassifiers_finished.rds'))
if(!methods::is(fit,'runClassifiers.obj')) stop('Unexpected class of runClassifiers RDS')
models <- methods::slot(fit,'models.info')
features <- methods::slot(fit,'model.features')
perfs <- methods::slot(fit,'performances')
preds <- methods::slot(fit,'predictions')
if(length(models)!=10L || length(features)!=10L) stop('Native object does not include all 10 entries')
summary <- vector('list',length(features))
for(i in seq_along(features)){
    item <- features[[i]]
    vals <- if(is.numeric(item)) item else if(is.data.frame(item)) unlist(item[vapply(item,is.numeric,logical(1L))],use.names=FALSE) else numeric(0)
    summary[[i]] <- data.frame(model_position=i,
      name=if(length(names(features))>=i) names(features)[i] else paste0('model_',i),
      feature_object_class=paste(class(item),collapse='|'),
      feature_element_count=length(item),
      directly_numeric_scores=length(vals),
      finite_scores=sum(is.finite(vals)),nonfinite_scores=sum(!is.finite(vals)),
      nonzero_finite_scores=sum(is.finite(vals)&vals!=0),
      needs_deeper_feature_shape_inspection=!is.numeric(item) && !is.data.frame(item))
}
utils::write.csv(do.call(rbind,summary),file.path(dst,'native_model_feature_inventory.csv'),row.names=FALSE)
util <- c('TENTACLES native read-only model-quality inspection',
          paste('source=',src),paste('models=',length(models)),
          paste('feature_lists=',length(features)),paste('consensus_genes=',length(gene)),
          paste('performance_object_class=',paste(class(perfs),collapse='|')),
          paste('predictions_rows=',if(is.data.frame(preds)) nrow(preds) else 'not_data_frame'),
          'WARNING: importance shape may need model-specific parsing; do not infer healthy fits just from list lengths')
for(i in seq_along(models)){
    util <- c(util,paste0('MODEL ',i,' ',if(length(names(models))>=i)names(models)[i] else 'unnamed',': ',
       paste(utils::capture.output(utils::str(models[[i]],max.level=1L,list.len=4L,give.attr=FALSE)),collapse=' | ')))
}
util <- c(util,'PERFORMANCES:',paste(utils::capture.output(utils::str(perfs,max.level=2L,list.len=10L,give.attr=FALSE)),collapse='\n'))
writeLines(substr(util,1L,3000L),file.path(dst,'native_model_structure_and_performance.txt'))
if(is.data.frame(preds)){
    utils::write.csv(data.frame(column=names(preds),class=vapply(preds,function(v)paste(class(v),collapse='|'),character(1))),file.path(dst,'native_training_prediction_schema.csv'),row.names=FALSE)
}
warnfile=file.path(src,'classifiers.log')
if(file.exists(warnfile)){
    w=readLines(warnfile,warn=FALSE)
    k <- c(mlp_too_many_weights=sum(grepl('too many \\(',w)),
       lightgbm_fraction_invalid=sum(grepl('feature_fraction_bynode',w,fixed=TRUE)),
       boruta_importance_NA=sum(grepl('getImp result contains NA',w,fixed=TRUE)),
       xgboost_early_stopping=sum(grepl('early.stopping|early_stopping',w,ignore.case=TRUE)))
    utils::write.csv(data.frame(category=names(k),occurrences=as.integer(k)),file.path(dst,'native_log_warnings.csv'),row.names=FALSE)
}
writeLines('PASS_CONSENSUS_VOTE_INTEGRITY; CLASSIFIER_QUALITY_REQUIRES_MANUAL_SCORE_INTERPRETATION',file.path(dst,'AUDIT_STATUS.txt'))
cat('PASS_NATIVE_READONLY_AUDIT=',dst,'\n',sep='')
