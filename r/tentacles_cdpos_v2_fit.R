#!/usr/bin/env Rscript
# CMPS355 native TENTACLES LOCO, version 2, CD-positive tuning.
# A transparent adaptation using the pinned official package, not a fork.
# FAIL CLOSED: a cheap actual-native smoke fit must prove CD-positive metrics.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=6L) stop('Usage: Rscript tentacles_cdpos_v2_fit.R MODE(smoke|fit) COUNTS CLINICAL OUT FOLD PINNED_SHA')
mode <- args[[1L]]; if(!mode %in% c('smoke','fit')) stop('Invalid mode')
counts_path <- normalizePath(args[[2L]], mustWork=TRUE)
clinical_path <- normalizePath(args[[3L]], mustWork=TRUE)
out <- normalizePath(args[[4L]], mustWork=TRUE)
fold <- args[[5L]]; if(!fold %in% paste0('d',1:4)) stop('Invalid fold')
expected_sha <- tolower(args[[6L]])
lib <- Sys.getenv('CMPS355_R_LIB','')
if(nzchar(lib) && dir.exists(lib)) .libPaths(c(lib,.libPaths()))
if(!requireNamespace('TENTACLES',quietly=TRUE)) stop('Missing official TENTACLES R package')
if(!requireNamespace('yardstick',quietly=TRUE)) stop('Missing yardstick')
if(as.character(utils::packageVersion('TENTACLES')) != '0.1.0') stop('Official package version mismatch')
remote_sha <- tolower(utils::packageDescription('TENTACLES')[['RemoteSha']])
if(is.null(remote_sha) || is.na(remote_sha) || remote_sha != expected_sha) stop('Official TENTACLES source commit differs from pinned revision')
# Explicit factor levels, not the defunct `yardstick.event_first` global option.
probe_truth <- factor(c('CD','Not IBD','CD','Not IBD'), levels=c('CD','Not IBD'))
probe_pred  <- factor(c('CD','CD','CD','Not IBD'), levels=c('CD','Not IBD'))
if(abs(as.numeric(yardstick::f_meas_vec(probe_truth,probe_pred))-0.8)>1e-12) stop('yardstick default event must be the first factor level')
if(!is.null(getOption('yardstick.event_first'))) stop('Remove obsolete yardstick.event_first option before fitting')
counts <- utils::read.csv(counts_path, check.names=FALSE, stringsAsFactors=FALSE)
meta <- utils::read.csv(clinical_path, check.names=FALSE, stringsAsFactors=FALSE)
if(!all(c('ID','Study','class','cohort') %in% names(meta)) || !'ID' %in% names(counts)) stop('Missing input metadata or counts ID')
if(!identical(as.character(meta$ID), as.character(counts$ID)) || anyDuplicated(meta$ID) || anyNA(meta$ID)) stop('Sample IDs not aligned/unique')
if(fold %in% unique(meta$cohort) || length(unique(meta$cohort))!=3L) stop('Outer holdout leaked into native training')
if(!setequal(unique(meta$class),c('CD','Not IBD'))) stop('Two expected outcome classes not present')
expected_n <- c(d1=429L,d2=510L,d3=613L,d4=515L)
if(nrow(counts)!=expected_n[[fold]]) stop('Unexpected number of training patients')
row.names(counts) <- as.character(counts$ID); counts$ID <- NULL
if(ncol(counts)!=9983L || anyDuplicated(names(counts)) || any(!vapply(counts,is.numeric,logical(1L)))) stop('Unexpected raw common-gene matrix')
if(anyNA(counts) || any(!is.finite(as.matrix(counts))) || any(as.matrix(counts)<0) || any(rowSums(as.matrix(counts))<=0)) stop('Invalid count data')
row.names(meta) <- as.character(meta$ID)
clin <- data.frame(class=factor(as.character(meta$class),levels=c('CD','Not IBD')),row.names=row.names(meta))
if(!identical(levels(clin$class),c('CD','Not IBD'))) stop('CD must be first class level')

# IMPORTANT: the pinned authors' preProcess() recodes labels to binary:
# "Not IBD" -> 0, "CD" -> 1 (as shown in its own log).  The V2 draft
# incorrectly searched only for literal "CD"/"Not IBD" inside its S4 slots.
# Keep the package's *values* unchanged, but reorder the factor levels so its
# actual event is 1=CD.  Verify this on both samples and native F1 after fitting.
relevel_outcomes <- function(object) {
  diagnostic <- character()
  changed <- character()
  valid_levels <- function(z, path) {
    zz <- as.character(z)
    if (!length(zz) || anyNA(zz)) stop('Missing class values at ',path)
    seen <- sort(unique(zz))
    if (identical(seen, sort(c('0','1')))) return(c('1','0'))
    if (identical(seen, sort(c('CD','Not IBD')))) return(c('CD','Not IBD'))
    stop('Unrecognized class coding at ',path,': ',paste(seen,collapse='|'),
         '. Refusing to relabel unknown values')
  }
  rec <- function(x,path,depth=0L) {
    if (depth>10L) stop('Unexpectedly nested preProcess object at ',path)
    if (is.data.frame(x)) {
      if ('class' %in% names(x)) {
        lev <- valid_levels(x[['class']],paste0(path,'$class'))
        diagnostic <<- c(diagnostic,paste0(path,'$class: ',
                paste(sort(unique(as.character(x[['class']]))),collapse='|'),
                ' => ',paste(lev,collapse='|')))
        x[['class']] <- factor(as.character(x[['class']]),levels=lev)
        changed <<- c(changed,paste0(path,'$class'))
      }
      return(x)
    }
    if (is.list(x)) {
      for (j in seq_along(x)) {
        nm <- names(x)[j]
        if (is.null(nm) || is.na(nm) || !nzchar(nm)) nm <- as.character(j)
        q <- paste0(path,'$',nm)
        if (identical(nm,'class') && (is.factor(x[[j]]) || is.atomic(x[[j]]))) {
          lev <- valid_levels(x[[j]],q)
          diagnostic <<- c(diagnostic,paste0(q,': ',
                paste(sort(unique(as.character(x[[j]]))),collapse='|'),
                ' => ',paste(lev,collapse='|')))
          x[[j]] <- factor(as.character(x[[j]]),levels=lev)
          changed <<- c(changed,q)
        } else x[[j]] <- rec(x[[j]],q,depth+1L)
      }
    }
    x
  }
  stopifnot(isS4(object),inherits(object,'preProcess.obj'))
  for (sn in intersect(c('raw','processed','metadata'),methods::slotNames(object))) {
    object <- methods::`slot<-`(object,sn,value=rec(methods::slot(object,sn),paste0('@',sn)))
  }
  if (!length(changed)) stop('No outcome columns found in preProcess.obj. ',
        'Inspect native_preprocess_structure_BEFORE_relevel.txt; cannot verify CD-first tuning')
  m <- methods::slot(object,'metadata')
  if (!is.data.frame(m) || !'class' %in% names(m) || !is.factor(m$class))
    stop('Native preProcess@metadata$class is absent or not a factor')
  expected_levels <- valid_levels(m$class,'@metadata$class')
  if (!identical(levels(m$class),expected_levels))
    stop('Native metadata outcome levels incorrectly ordered')
  # The accepted input has both classes in each discovery fold.  Verify the
  # original clinical outcome prevalence against the binary metadata coding.
  meta_cd <- sum(as.character(clin$class)=='CD')
  native_cd <- sum(as.character(m$class)==expected_levels[[1L]])
  if (nrow(m)!=nrow(clin) || meta_cd!=native_cd)
    stop('Native metadata counts do not match original CD discovery labels')
  if (setequal(rownames(m),rownames(clin))) {
    original <- as.character(clin[rownames(m),'class'])
    native <- as.character(m$class)
    if (!identical(original=='CD',native==expected_levels[[1L]]))
      stop('Native metadata patient identities do not match clinical labels')
    cat('NATIVE_PREPROCESS_PATIENT_LABEL_ALIGNMENT=TRUE\n')
  } else {
    cat('NATIVE_PREPROCESS_PATIENT_ID_ORDER_NOT_EXPOSED: original input ID ',
        'matching was verified separately; native label count was checked\n',sep='')
  }
  writeLines(c(diagnostic,paste0('CHANGED=',paste(changed,collapse=','))),
             file.path(out,'preprocess_event_level_audit.txt'))
  cat('NATIVE_METADATA_CLASS_LEVELS=',paste(levels(m$class),collapse='|'),'\n',sep='')
  cat('CD_FIRST_COLUMNS_RELEVELED=',length(changed),'\n',sep='')
  object
}

cpm_preprocess <- function(mat) {
  p <- TENTACLES::preProcess(mat,clin,class='class',case.label='CD',
          mincpm=1,minfraction=0.1,data.type='rnaseq',is.normalized=FALSE,
          batch=NULL,covar.mod=NULL,plot=FALSE)
  if(!inherits(p,'preProcess.obj')) stop('Native preProcess did not yield preProcess.obj')
  writeLines(capture.output(str(p,max.level=3)),
             file.path(out,'native_preprocess_structure_BEFORE_relevel.txt'))
  relevel_outcomes(p)
}

# The authors' native predictions can use 0/1 even when the source
# clinical labels were CD/Not IBD.  1 means CD because case.label='CD'.
canonical_class <- function(z,where) {
  values <- as.character(z)
  if (!length(values) || anyNA(values)) stop('Empty/missing predictions: ',where)
  seen <- sort(unique(values))
  if (all(seen %in% c('0','1'))) return(ifelse(values=='1','CD','Not IBD'))
  if (all(seen %in% c('CD','Not IBD'))) return(values)
  stop('Unexpected class representation at ',where,': ',paste(seen,collapse='|'))
}
f1_calc <- function(truth,pred,positive) {
  tp <- sum(truth==positive & pred==positive)
  fp <- sum(truth!=positive & pred==positive)
  fn <- sum(truth==positive & pred!=positive)
  if (2L*tp+fp+fn == 0L) return(NA_real_)
  2*tp/(2*tp+fp+fn)
}

# Native performs apparent metrics on the pooled discovery rows. Verify actual
# f_meas event class via its stored predictions, not factor levels alone.
metric_orientation <- function(fit, required_models=1L) {
  stopifnot(isS4(fit),inherits(fit,'runClassifiers.obj'))
  pred <- methods::slot(fit,'predictions')
  per <- methods::slot(fit,'performances')
  final <- per[['final_metrics']]
  tune <- per[['tuning_metrics']]
  if(!is.data.frame(pred) || !all(c('.pred_class','class','model') %in% names(pred))) stop('Native training prediction schema unexpected')
  if(!is.data.frame(final) || !all(c('model','.metric','.estimate') %in% names(final))) stop('Native final metric schema unexpected')
  if(!is.data.frame(tune) || !all(c('wflow_id','.metric','mean','n') %in% names(tune))) stop('Native CV metric schema unexpected')
  models <- unique(as.character(pred$model))
  if(length(models)!=required_models) stop('Wrong number of native predictions-model groups')
  rows <- list(); mismatched <- character(); identified <- 0L
  for(m in models) {
    p <- pred[as.character(pred$model)==m,,drop=FALSE]
    if(nrow(p)!=nrow(meta)) stop('Invalid fitted-development prediction count: ',m)
    truth <- canonical_class(p$class,paste0(m,'$class'))
    yhat <- canonical_class(p$.pred_class,paste0(m,'$.pred_class'))
    if (sum(truth=='CD')!=sum(as.character(meta$class)=='CD'))
      stop('Development truth class prevalence inconsistent with original metadata: ',m)
    fcd <- f1_calc(truth,yhat,'CD'); fctrl <- f1_calc(truth,yhat,'Not IBD')
    reported <- final[as.character(final$model)==m & as.character(final$.metric)=='f_meas','.estimate',drop=TRUE]
    if(length(reported)!=1L || !is.finite(reported)) stop('Missing/invalid native fitted F1 for model ',m)
    cdm <- is.finite(fcd) && abs(fcd-reported)<1e-8
    ctrlm <- is.finite(fctrl) && abs(fctrl-reported)<1e-8
    if(!cdm && ctrlm) mismatched <- c(mismatched,m)
    if(!cdm && !ctrlm) stop('Native final F1 matches neither class; model=',m)
    if(cdm && !ctrlm) identified <- identified+1L
    rows[[length(rows)+1L]] <- data.frame(model=m,f1_native=reported,
      f1_CD=fcd,f1_Not_IBD=fctrl,matches_CD=cdm,matches_Not_IBD=ctrlm,
      identifiable=cdm!=ctrlm,stringsAsFactors=FALSE)
  }
  audit <- do.call(rbind,rows)
  utils::write.csv(audit,file.path(out,'native_f1_event_orientation.csv'),row.names=FALSE)
  if(length(mismatched)) stop('Native metric is still control-positive for models: ',paste(mismatched,collapse=','))
  if(identified<1L) stop('Native metric event class ambiguous: all models have equal class F1; no sufficient evidence')
  ff <- tune[tune$.metric=='f_meas',,drop=FALSE]
  if(nrow(ff)!=required_models || any(!is.finite(ff$mean)) || any(ff$n!=if(mode=='smoke') 2L else 5L) ||
      length(unique(as.character(ff$wflow_id)))!=required_models) {
    stop('Native tuning F1 missing/invalid/incomplete (expected one selected config per model)')
  }
  utils::write.csv(ff,file.path(out,'native_selected_cv_f1.csv'),row.names=FALSE)
  cat('CD_POSITIVE_F1_VERIFIED=',identified,'/',required_models,';AMBIGUOUS=',required_models-identified,'\n',sep='')
  invisible(audit)
}

Sys.setenv(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
options(warn=1)
set.seed(123L)
suppressPackageStartupMessages(library('TENTACLES'))
writeLines(c(paste0('MODE=',mode),paste0('HELDOUT=',fold),
 'CASE_LABEL=CD','INTERNAL_CLASS_ORDER=1|0 (CD|Not IBD)','EVENT_LEVEL=first',
 'TENTACLES_SOURCE_SHA=069a26271e3885e015df25baa40a7a1140827d5f',
 'MODEL_TUNING=tune_grid;F1'),file.path(out,'protocol_cd_positive_v2.txt'))
models <- c('rand_forest','mlp','bag_mlp','svm_linear','C5_rules','lightGBM','decision_tree','mars','xgboost','pls')
selectors <- c('corr','roc','roc','boruta','boruta','roc','boruta','boruta','base','boruta')
if(mode=='smoke') {
  # Only discovery samples; intentionally modest gene subset for an actual quick native fit.
  abundance <- colSums(as.matrix(counts))
  ok <- names(abundance)[is.finite(abundance) & abundance>0]
  if(length(ok)<16L) stop('Too few expressed genes for smoke fit')
  small <- ok[order(abundance[ok],decreasing=TRUE)][seq_len(min(24L,length(ok)))]
  p <- cpm_preprocess(counts[,small,drop=FALSE])
  writeLines(capture.output(str(p,max.level=2)),file.path(out,'native_preprocess_smoke_structure.txt'))
  cat('SMOKE_NATIVE_START\n')
  fitted <- TENTACLES::runClassifiers(preProcess.obj=p,models='decision_tree',
    selector.recipes='base',tuning.method='tune_grid',n=2,v=2,
    boruta.maxRuns=100,selector.threshold=0.95,metric='f_meas',
    nsim=2,filter=FALSE,seed=123,downsample=TRUE,plot=FALSE,parallel=FALSE)
  metric_orientation(fitted,required_models=1L)
  writeLines(capture.output(utils::sessionInfo()),file.path(out,'r_session_info.txt'))
  writeLines('PASS_SMOKE_NATIVE_CD_POSITIVE_F1',file.path(out,'SMOKE_PASS.txt'))
  cat('SMOKE_PASS_CD_POSITIVE_F1=TRUE\n')
} else {
  if(file.exists(file.path(out,'runclassifiers_finished.rds'))) stop('Will not overwrite a completed native object')
  p <- cpm_preprocess(counts)
  saveRDS(p,file.path(out,'train_only_preprocessed.rds'),compress='gzip')
  utils::write.csv(data.frame(model=models,selector=selectors),file.path(out,'model_selector_pairs.csv'),row.names=FALSE)
  cat('FULL_NATIVE_FIT_CD_POSITIVE_START\n')
  st <- proc.time()[[3L]]
  fitted <- TENTACLES::runClassifiers(preProcess.obj=p,models=models,
    selector.recipes=selectors,tuning.method='tune_grid',n=5,v=5,
    boruta.maxRuns=100,selector.threshold=0.95,metric='f_meas',
    nsim=2,filter=TRUE,seed=123,downsample=TRUE,plot=FALSE,parallel=FALSE)
  if(is.null(fitted) || !inherits(fitted,'runClassifiers.obj')) stop('Official native classifier fit did not return its object')
  # Save first: preserve many-hour fitting evidence even if quality check below fails.
  partial <- file.path(out,'runclassifiers_finished.rds.partial')
  saveRDS(fitted,partial,compress='gzip')
  if(!file.rename(partial,file.path(out,'runclassifiers_finished.rds'))) stop('Atomic native RDS finalization failed')
  writeLines(capture.output(str(fitted,max.level=2)),file.path(out,'runclassifiers_structure.txt'))
  feature_sets <- methods::slot(fitted,'model.features')
  if(!is.list(feature_sets) || length(feature_sets)!=10L) stop('Expected ten official classifier feature sets')
  feature_rows <- list()
  for(j in seq_along(feature_sets)) {
    ff <- feature_sets[[j]]
    if(!is.data.frame(ff) || !all(c('Variable','Importance') %in% names(ff))) stop('Unexpected native VIP feature table in model ',j)
    importance <- suppressWarnings(as.numeric(ff[['Importance']]))
    nm <- as.character(ff[['Variable']])
    if(length(importance)!=nrow(ff) || anyNA(importance) || any(!is.finite(importance)) ||
       anyNA(nm) || any(!nzchar(nm)) || anyDuplicated(nm)) stop('Invalid variable importance in model ',j)
    if(!all(nm %in% names(counts))) stop('Native feature names not mapped to accepted original gene symbols, model ',j)
    if(!any(importance!=0)) stop('Entire native model feature importance is zero: model ',j)
    feature_rows[[j]] <- data.frame(model=names(feature_sets)[j],n_native_features=length(nm),
            n_nonzero=sum(importance!=0),min_importance=min(importance),max_importance=max(importance))
  }
  utils::write.csv(do.call(rbind,feature_rows),file.path(out,'native_feature_importance_quality.csv'),row.names=FALSE)
  metric_orientation(fitted,required_models=10L)
  writeLines(capture.output(utils::sessionInfo()),file.path(out,'r_session_info.txt'))
  ip<-as.data.frame(utils::installed.packages()[,c('Package','Version'),drop=FALSE]); utils::write.csv(ip,file.path(out,'installed_R_package_versions.csv'),row.names=FALSE)
  saveRDS(methods::slot(fitted,'model.features'),file.path(out,'model_features_native.rds'))
  saveRDS(methods::slot(fitted,'performances'),file.path(out,'performances_native.rds'))
  cat('FULL_NATIVE_ELAPSED_SECONDS=',sprintf('%.1f',proc.time()[[3L]]-st),'\n',sep='')
  cat('CLASSIFIERS_PASS_CD_POSITIVE=TRUE\n')
}
