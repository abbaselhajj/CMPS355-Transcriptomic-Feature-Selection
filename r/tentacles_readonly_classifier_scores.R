#!/usr/bin/env Rscript
# Read-only extraction of official native cross-validation and apparent metrics.
# Uses small performances_native.rds snapshots; does not re-fit classifiers.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop('Usage: Rscript tentacles_readonly_classifier_scores.R PROJECT_ROOT OUTPUT_DIR')
root <- normalizePath(args[[1]], mustWork=TRUE)
out <- file.path(normalizePath(dirname(args[[2]]),mustWork=TRUE), basename(args[[2]]))
if(dir.exists(out) && length(list.files(out,all.files=TRUE,no..=TRUE))>0L) stop('Refusing to overwrite an audit directory')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
all <- list(); issues <- character()
for (fold in c('d1','d2','d3','d4')) {
  dirs <- list.dirs(file.path(root,'runs','tentacles_stage9_loco',fold), recursive=FALSE,full.names=TRUE)
  valid <- dirs[vapply(dirs,function(d){
    s <- file.path(d,'STATUS.txt')
    file.exists(s) && identical(trimws(readLines(s,n=1L,warn=FALSE)),'PASS_NATIVE_CLASSIFIERS_AND_CONSENSUS') && file.exists(file.path(d,'performances_native.rds'))
  },logical(1L))]
  if (length(valid)!=1L) stop('Expected exactly one completed performance snapshot for ',fold,'; found ',length(valid))
  p <- readRDS(file.path(valid,'performances_native.rds'))
  if(!is.list(p)||!all(c('tuning_metrics','final_metrics')%in%names(p))) stop('Unexpected performances structure: ',fold)
  tt <- as.data.frame(p$tuning_metrics); ff <- as.data.frame(p$final_metrics)
  if(nrow(tt)!=40L || nrow(ff)!=40L) stop(fold, ': unexpected metric rows')
  if(!all(c('wflow_id','.metric','mean')%in%names(tt))) stop(fold, ': tuning metrics columns changed')
  tt$heldout <- fold
  ff$heldout <- fold
  utils::write.csv(tt,file.path(out,paste0(fold,'_tuning_metrics.csv')),row.names=FALSE,na='')
  utils::write.csv(ff,file.path(out,paste0(fold,'_final_metrics.csv')),row.names=FALSE,na='')
  f1 <- tt[tt$.metric=='f_meas',c('wflow_id','mean','std_err','n'),drop=FALSE]
  if(nrow(f1)!=10L || anyDuplicated(f1$wflow_id)) stop(fold, ': F1 score coverage is incomplete')
  f1$heldout <- fold
  f1$cv_f1_finite <- is.finite(f1$mean)
  f1$cv_f1_bounded <- is.finite(f1$mean) & f1$mean >= 0 & f1$mean <= 1
  f1$warning_scope <- ifelse(grepl('^(mlp_|lightGBM_)',f1$wflow_id), 'see warning log; score validity requires interpretation', 'none attributable from this table')
  all[[fold]] <- f1[,c('heldout','wflow_id','mean','std_err','n','cv_f1_finite','cv_f1_bounded','warning_scope')]
}
all <- do.call(rbind,all)
utils::write.csv(all,file.path(out,'all_models_crossvalidated_f1.csv'),row.names=FALSE,na='')
summary <- data.frame(heldout=c('d1','d2','d3','d4'),n_models=10L,
 n_cv_f1_finite=sapply(c('d1','d2','d3','d4'),function(f)sum(all$heldout==f & all$cv_f1_finite)),
 n_cv_f1_bounded=sapply(c('d1','d2','d3','d4'),function(f)sum(all$heldout==f & all$cv_f1_bounded)))
utils::write.csv(summary,file.path(out,'crossvalidated_score_quality_summary.csv'),row.names=FALSE)
if(any(!all$cv_f1_bounded)){
 writeLines('REVIEW_REQUIRED: one or more F1 estimates missing or outside [0,1]; inspect tuning metrics',file.path(out,'AUDIT_STATUS.txt'))
} else {
 writeLines('PASS_CV_F1_FINITE_ALL_MODELS; WARNING_EFFECTS_AND_FINAL_FITS_REQUIRE_INTERPRETATION',file.path(out,'AUDIT_STATUS.txt'))
}
cat('READONLY_F1_EXTRACTION_COMPLETE=',out,'\n',sep='')
print(summary)
