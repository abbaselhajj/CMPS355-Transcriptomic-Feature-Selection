#!/usr/bin/env Rscript
# Strict, additive D4-only audit and conditional nine-voter consensus recovery.
# Original native fold, STATUS, logs, and fitted .rds are NEVER modified.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop('Usage: Rscript recover_d4_cv_gate.R ORIGINAL_D4_DIR NEW_RECOVERY_DIR')
src <- normalizePath(args[[1L]], mustWork=TRUE)
out <- normalizePath(args[[2L]], mustWork=TRUE)
root <- normalizePath(file.path(src,'../../../../'), mustWork=TRUE)
if (basename(dirname(src))!='d4' ||
    !grepl('/runs/tentacles_cd_positive_v2/d4/',paste0(src,'/'),fixed=TRUE)) stop('Unexpected original D4 path')
if (!grepl('/runs/tentacles_cd_positive_v2_recovery/d4/',paste0(out,'/'),fixed=TRUE)) stop('Unexpected D4 recovery path')
if (identical(src,out)) stop('Original and destination must differ')
if (file.exists(file.path(src,'consensus_native_finished.rds'))) stop('Original already has consensus; require manual review')
fit_path <- file.path(src,'runclassifiers_finished.rds')
if (!file.exists(fit_path) || file.info(fit_path)$size < 1000000) stop('Saved native fit absent/too small; stop')
if (!file.exists(file.path(src,'classifiers.log')) ||
    !any(grepl('Native tuning F1 missing/invalid/incomplete',readLines(file.path(src,'classifiers.log'),warn=FALSE),fixed=TRUE))) {
  stop('This is not the expected CV quality-gate failure; no automatic recovery')
}
lib <- Sys.getenv('CMPS355_R_LIB','')
if(nzchar(lib) && dir.exists(lib)) .libPaths(c(lib,.libPaths()))
if (!requireNamespace('TENTACLES',quietly=TRUE)) stop('TENTACLES unavailable')
if (as.character(getRversion())!='4.6.1') stop('Wrong R version')
if (as.character(utils::packageVersion('TENTACLES'))!='0.1.0') stop('Wrong native package version')
sha <- tolower(utils::packageDescription('TENTACLES')[['RemoteSha']])
if (length(sha)!=1L || is.na(sha) || sha!='069a26271e3885e015df25baa40a7a1140827d5f') stop('Pinned native source differs')
options(warn=1)
expected <- c('rand_forest_corr','mlp_roc','bag_mlp_roc','svm_linear_boruta',
              'C5_rules_boruta','lightGBM_roc','decision_tree_boruta',
              'mars_boruta','pls_boruta')
xgb <- 'xgboost_base'
all_models <- c(expected,xgb)
cat('D4_SAVED_NATIVE_FIT_READ_START\n')
fit <- readRDS(fit_path)
if(!isS4(fit) || !inherits(fit,'runClassifiers.obj')) stop('Unexpected native fit object')
per <- methods::slot(fit,'performances')
tune <- per[['tuning_metrics']]
fin <- per[['final_metrics']]
pred <- methods::slot(fit,'predictions')
fs <- methods::slot(fit,'model.features')
if(!is.data.frame(tune) || !all(c('wflow_id','.metric','mean','n') %in% names(tune))) stop('Unexpected CV table')
if(!is.data.frame(fin) || !all(c('model','.metric','.estimate') %in% names(fin))) stop('Unexpected fitted metric table')
if(!is.data.frame(pred) || !all(c('model','class','.pred_class') %in% names(pred))) stop('Unexpected fitted predictions')
if(!is.list(fs) || length(fs)!=10L || is.null(names(fs)) || anyDuplicated(names(fs)) ||
   !setequal(names(fs),all_models)) stop('Need all ten original nonmissing model feature-table names')
# Always publish the actual native CV rows, including bad and duplicate records.
utils::write.csv(tune,file.path(out,'native_all_tuning_metrics.csv'),row.names=FALSE)
ff <- tune[!is.na(tune$.metric) & as.character(tune$.metric)=='f_meas',,drop=FALSE]
utils::write.csv(ff,file.path(out,'native_all_f1_rows.csv'),row.names=FALSE)
quality <- list()
for (m in all_models) {
  a <- ff[!is.na(ff$wflow_id) & as.character(ff$wflow_id)==m,,drop=FALSE]
  good <- nrow(a)==1L && is.numeric(a$mean) && is.finite(a$mean[[1L]]) &&
          is.numeric(a$n) && is.finite(a$n[[1L]]) && a$n[[1L]]==5L
  quality[[m]] <- data.frame(model=m,rows=nrow(a),
                            n_configurations=if('.config'%in%names(a))length(unique(as.character(a$.config))) else NA_integer_,
                            cv_f1=if(nrow(a)==1L) as.character(a$mean[[1L]]) else NA_character_,
                            cv_folds=if(nrow(a)==1L) as.character(a$n[[1L]]) else NA_character_,
                            valid_single_selected_f1=good,stringsAsFactors=FALSE)
}
q <- do.call(rbind,quality)
utils::write.csv(q,file.path(out,'native_cv_f1_by_model.csv'),row.names=FALSE)
cat('D4_NATIVE_CV_F1_AUDIT\n'); print(q,row.names=FALSE)
# Require the CD-positive apparent fitted score to agree with independently recomputed F1.
f1 <- function(y,yh,event) {
  tp <- sum(y==event & yh==event)
  fp <- sum(y!=event & yh==event)
  fn <- sum(y==event & yh!=event)
  if(2*tp+fp+fn==0L) return(NA_real_)
  2*tp/(2*tp+fp+fn)
}
# Do not infer the identity of the positive class from string order.
# TENTACLES preProcess explicitly documents Not IBD=0 and CD=1 for this project.
# The original fitting script relevelled CD first. Accept only those two known
# storage encodings and audit both before and after normalisation.
canon <- function(x) {
  z <- as.character(x)
  if (!length(z) || anyNA(z)) return(list(ok=FALSE, value=character(), encoding='MISSING'))
  if (all(z %in% c('CD','Not IBD'))) return(list(ok=TRUE, value=z, encoding='NAMED'))
  if (all(z %in% c('1','0'))) return(list(ok=TRUE, value=ifelse(z=='1','CD','Not IBD'), encoding='NATIVE_1_CD_0_NOT_IBD'))
  list(ok=FALSE,value=character(),encoding='UNEXPECTED')
}
ma <- list()
for(m in all_models) {
  p <- pred[!is.na(pred$model) & as.character(pred$model)==m,,drop=FALSE]
  raw_y <- as.character(p$class)
  raw_yh <- as.character(p$.pred_class)
  tr <- canon(raw_y); pr <- canon(raw_yh)
  a <- fin[!is.na(fin$model) & !is.na(fin$.metric) &
           as.character(fin$model)==m & as.character(fin$.metric)=='f_meas','.estimate',drop=TRUE]
  sufficient <- nrow(p)==515L && tr$ok && pr$ok &&
    setequal(unique(tr$value),c('CD','Not IBD'))
  cd <- if(sufficient) f1(tr$value,pr$value,'CD') else NA_real_
  ct <- if(sufficient) f1(tr$value,pr$value,'Not IBD') else NA_real_
  matches_cd <- sufficient && length(a)==1L && is.finite(a) &&
    is.finite(cd) && abs(cd-a)<1e-8
  matches_control <- sufficient && length(a)==1L && is.finite(a) &&
    is.finite(ct) && abs(ct-a)<1e-8
  okay <- matches_cd && !matches_control
  failure_reason <- if (nrow(p)!=515L) 'WRONG_ROW_COUNT' else
    if (!tr$ok || !pr$ok) 'UNEXPECTED_NATIVE_CLASS_CODING' else
    if (!setequal(unique(tr$value),c('CD','Not IBD'))) 'TRUTH_MISSING_CLASS' else
    if (length(a)!=1L || !is.finite(a)) 'MISSING_OR_INVALID_REPORTED_F1' else
    if (!is.finite(cd)) 'UNDEFINED_INDEPENDENT_CD_F1' else
    if (!matches_cd) 'REPORTED_F1_DOES_NOT_MATCH_CD' else
    if (matches_control) 'CLASS_F1_ORIENTATION_AMBIGUOUS' else 'PASS'
  ma[[m]] <- data.frame(model=m, training_rows=nrow(p),
    truth_encoding=tr$encoding, prediction_encoding=pr$encoding,
    truth_raw_values=paste(sort(unique(raw_y)),collapse='|'),
    predictions_raw_values=paste(sort(unique(raw_yh)),collapse='|'),
    truth_cd_count=if(sufficient)sum(tr$value=='CD') else NA_integer_,
    truth_control_count=if(sufficient)sum(tr$value=='Not IBD') else NA_integer_,
    reported_f1=if(length(a)==1L)as.character(a) else NA_character_,
    independently_cd_f1=cd,not_ibd_f1=ct,
    reported_matches_cd=matches_cd,reported_matches_not_ibd=matches_control,
    cd_positive_identified=okay,failure_reason=failure_reason,
    stringsAsFactors=FALSE)
}
ma <- do.call(rbind,ma)
utils::write.csv(ma,file.path(out,'native_f1_orientation_readback.csv'),row.names=FALSE)
cat('D4_F1_ORIENTATION_DIAGNOSTIC_V2\n')
print(ma[,c('model','truth_encoding','prediction_encoding',
            'independently_cd_f1','not_ibd_f1',
            'cd_positive_identified','failure_reason')],row.names=FALSE)
fr<-list()
for(m in all_models) {
  f <- fs[[m]]
  schema <- is.data.frame(f) && all(c('Variable','Importance') %in% names(f))
  nm <- if(schema) as.character(f$Variable) else character()
  imp <- if(schema) suppressWarnings(as.numeric(f$Importance)) else numeric()
  good <- schema && length(nm)>0L && length(imp)==length(nm) &&
          !anyNA(nm) && all(nzchar(nm)) && !anyDuplicated(nm) &&
          !anyNA(imp) && all(is.finite(imp)) && any(imp!=0)
  fr[[m]] <- data.frame(model=m,n_features=length(nm),n_nonzero=sum(imp!=0,na.rm=TRUE),feature_table_valid=good)
}
fr<-do.call(rbind,fr)
utils::write.csv(fr,file.path(out,'native_feature_importance_readback.csv'),row.names=FALSE)
problems <- c(
  if (!all(q$valid_single_selected_f1[q$model %in% expected])) 'At least one of NINE consensus voters lacks one complete, finite five-fold selected CV F1' else character(),
  if (!all(ma$cd_positive_identified[ma$model %in% expected])) 'At least one of NINE consensus voters lacks unambiguous CD-positive fitted F1' else character(),
  if (!all(fr$feature_table_valid[fr$model %in% expected])) 'At least one of NINE consensus voters lacks valid nonzero feature importance' else character()
)
if(length(problems)) {
  writeLines(c('REVIEW_REQUIRED_NINE_VOTER_GATE_FAILED',problems),file.path(out,'RECOVERY_STATUS.txt'))
  cat('STOP_D4_NINE_VOTER_GATE_NOT_VERIFIED\n',paste(problems,collapse='\n'),'\n')
  quit(save='no',status=3L)
}
# If ten-model CV looks valid on readback, there is a discrepancy to investigate
# rather than silently relabel this as an XGBoost-only problem.
if (all(q$valid_single_selected_f1) && all(ma$cd_positive_identified) && all(fr$feature_table_valid)) {
  writeLines('REVIEW_REQUIRED_ORIGINAL_GATE_DISAGREES_WITH_READBACK',file.path(out,'RECOVERY_STATUS.txt'))
  stop('All ten appear valid, inconsistent with original reported gate. Inspect native CV table before promotion.')
}
cat('D4_NINE_VOTERS_SELECTED_CV_F1_PASS=TRUE\n')
cat('D4_NINE_VOTERS_CD_POSITIVE_F1_PASS=TRUE\n')
cat('D4_NINE_VOTERS_FEATURE_IMPORTANCE_PASS=TRUE\n')
writeLines('NINE_VOTER_GATES_PASSED_CONSENSUS_RUNNING',file.path(out,'RECOVERY_STATUS.txt'))
# Official getConsensus on exactly the original unmodified fitted object; the XGBoost
# model is excluded from the predeclared consensus regardless of its CV status.
run_consensus <- function(nmin) {
  tag <- paste0('nmin',nmin)
  cat('AUTHOR_GETCONSENSUS_',tag,'_START\n',sep='')
  set.seed(123L)
  result <- TENTACLES::getConsensus(fit,n.min=as.integer(nmin),exclude=xgb,plot=FALSE)
  if(!is.list(result) || !all(c('consensusGenes','dataFrame')%in%names(result))) stop('Native consensus returned unexpected structure')
  gene <- result$consensusGenes
  matrix <- as.matrix(result$dataFrame)
  if(!is.character(gene) || !length(gene) || anyNA(gene) || any(!nzchar(gene)) ||
     anyDuplicated(gene) || nrow(matrix)!=9L || is.null(colnames(matrix)) ||
     anyDuplicated(colnames(matrix)) || anyNA(matrix) || !all(matrix %in% c(0,1))) stop('Native consensus vote quality failed')
  vote <- colSums(matrix)
  deriv <- names(vote)[vote>=nmin]
  if (!setequal(gene,deriv) || length(gene)!=length(deriv)) stop('Native and independent vote sets disagree')
  utils::write.csv(data.frame(gene=gene),file.path(out,paste0('consensus_genes_',tag,'.csv')),row.names=FALSE)
  utils::write.csv(data.frame(gene=names(vote),votes=as.integer(vote)),file.path(out,paste0('native_votes_',tag,'.csv')),row.names=FALSE)
  cat('OFFICIAL_CONSENSUS_',tag,'_GENES=',length(gene),'\n',sep='')
  result
}
six <- run_consensus(6L)
saveRDS(six,file.path(out,'consensus_native_finished.rds'))
utils::write.csv(data.frame(gene=six$consensusGenes),file.path(out,'consensus_genes_native.csv'),row.names=FALSE)
g6 <- six$consensusGenes
rm(six); gc(verbose=FALSE)
five <- run_consensus(5L)
if (!all(g6 %in% five$consensusGenes)) stop('Six-vote set not nested in five-vote set')
saveRDS(five,file.path(out,'consensus_nmin5_finished.rds'))
writeLines(c('RECOVERY_PASS_OFFICIAL_NINE_VOTER_CONSENSUS',
 'ORIGINAL_FIT_UNMODIFIED=TRUE','NINE_VOTERS_CD_POSITIVE_F1_VALID=TRUE',
 'XGBOOST_SELECTED_CV_N=4_NOT_ACCEPTED_AS_FIVEFOLD=TRUE',
 'NINE_VOTERS_SELECTED_FIVEFOLD_CV_VALID=TRUE','NINE_VOTERS_FEATURE_IMPORTANCE_VALID=TRUE',
 'NMIN6_INDEPENDENT_VOTE_CHECK=TRUE','NMIN5_INDEPENDENT_VOTE_CHECK=TRUE',
 'XGBOOST_EXCLUDED_PREDEFINED=TRUE',
 'TEN_MODEL_NATIVE_GATE_ORIGINALLY_FAILED=TRUE',
 'DO_NOT_PROMOTE_UNTIL_REVIEW=TRUE'),file.path(out,'RECOVERY_STATUS.txt'))
cat('D4_RECOVERY_PASS_OFFICIAL_NINE_VOTER_CONSENSUS=TRUE\n')
