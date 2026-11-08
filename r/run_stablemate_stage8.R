#!/usr/bin/env Rscript
# Official StableMate LOCO adaptation. No reimplementation of its selector.
options(stringsAsFactors = FALSE)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop('Usage: Rscript run_stablemate_stage8.R INPUT_WORKDIR DRIVE_OUTPUT R_LIBRARY')
work <- normalizePath(args[[1L]], mustWork=TRUE)
out <- normalizePath(args[[2L]], mustWork=TRUE)
lib <- normalizePath(args[[3L]], mustWork=TRUE)
Sys.setenv(R_LIBS=lib,R_LIBS_USER=lib)
.libPaths(unique(c(lib,.libPaths())))
if (!requireNamespace('StableMate',quietly=TRUE)) stop('Official StableMate package not installed')
if (!requireNamespace('jsonlite',quietly=TRUE)) stop('jsonlite package missing')
if (!requireNamespace('foreach',quietly=TRUE) || !requireNamespace('doParallel',quietly=TRUE))
  stop('StableMate parallel dependencies missing')

expected_sha <- 'cf0fe9f344c1756b6f89fd33f9fdae1a7f35b24a'
metadata <- utils::packageDescription('StableMate')
actual_sha <- tolower(as.character(metadata[['RemoteSha']]))
if (!identical(actual_sha,expected_sha)) stop('Official source revision mismatch: ',actual_sha)
if (!identical(as.character(utils::packageVersion('StableMate')),'0.1.0')) stop('Official package version mismatch')
manifest <- jsonlite::read_json(file.path(out,'manifest.json'),simplifyVector=TRUE)
if (!identical(manifest$source_commit,expected_sha)) stop('Manifest source commit mismatch')
if (manifest$model$model_K != 2000L || manifest$model$prefilter_K != 2000L ||
    manifest$model$prefilter_p != 50L) stop('Unexpected K=2000 protocol settings')
if (manifest$model$ncore < 1L || manifest$model$ncore > 2L) stop('ncore must be 1 or 2')

ns <- asNamespace('StableMate')
required <- c('lasso_prefilt','stablemate','reg_logit','obj_bic_logit','obj_pss_logit','pred_logit')
missing <- required[!vapply(required,exists,logical(1),envir=ns,inherits=FALSE)]
if (length(missing)) stop('Missing required official functions: ',paste(missing,collapse=', '))
api <- file.path(out,'author_api_formals.txt')
con <- file(api,open='wt')
for (nm in required) {
  cat('FUNCTION=',nm,'\n',sep='',file=con)
  cat(paste(capture.output(print(formals(get(nm,envir=ns)))),collapse='\n'),'\n',file=con)
}
close(con)
cat('OFFICIAL_STABLEMATE_SHA=',actual_sha,'\n',sep='')
cat('OFFICIAL_STABLEMATE_VERSION=',as.character(utils::packageVersion('StableMate')),'\n',sep='')
cat('MANIFEST_RUN_KEY=',manifest$run_key,'\n',sep='')
cat('R_VERSION=',as.character(getRversion()),'\n',sep='')
cat('AUTHOR_API_VERIFIED=TRUE\n')

read_x <- function(split) {
  df <- utils::read.csv(file.path(work,paste0(split,'_x.csv')),check.names=FALSE,stringsAsFactors=FALSE)
  if (names(df)[1L] != 'ID' || anyDuplicated(names(df)) || anyNA(df$ID))
    stop('Invalid ',split,' feature matrix')
  X <- as.matrix(df[,-1L,drop=FALSE]); storage.mode(X) <- 'double'
  if (any(!is.finite(X)) || ncol(X) < 50L) stop('Non-finite/insufficient predictors: ',split)
  list(id=as.character(df$ID),X=X)
}
tr <- read_x('train');te <- read_x('test')
if (!identical(colnames(tr$X),colnames(te$X))) stop('Train/test gene mismatch')
if (anyDuplicated(c(tr$id,te$id))) stop('Patient ID overlap')
read_metadata <- function(split) {
  df <- utils::read.csv(file.path(work,paste0(split,'_meta.csv')),check.names=FALSE,stringsAsFactors=FALSE)
  if (!all(c('ID','class','y','cohort','Study') %in% names(df))) stop('Missing metadata fields')
  if (!identical(as.character(df$ID),if(split=='train')tr$id else te$id)) stop('Patient ID order mismatch')
  if (!all(df$y==as.integer(df$class=='CD'))) stop('Label encoding mismatch')
  df
}
mtrain <- read_metadata('train');mtest <- read_metadata('test')
ytrain <- as.integer(mtrain$y)
ytest <- as.integer(mtest$y)
env <- as.character(mtrain$cohort)
if (!identical(sort(unique(env)),sort(manifest$training_cohorts))) stop('Environment mismatch')
if (length(unique(env)) != 3L || length(unique(ytrain)) != 2L) stop('Invalid training cohort/label structure')
if (any(mtest$cohort %in% env) || any(mtest$cohort != manifest$heldout)) stop('Outer cohort leaked into training')
if (nrow(tr$X)!=manifest$train_patients || nrow(te$X)!=manifest$test_patients ||
    ncol(tr$X)!=manifest$eligible_genes) stop('Manifest dimensions mismatch')
set.seed(as.integer(manifest$model$seed))

save_atomic_rds <- function(value,path) {
  tmp <- paste0(path,'.tmp')
  saveRDS(value,tmp,compress=TRUE)
  if (!file.rename(tmp,path)) stop('Could not atomically publish RDS: ',path)
}
load_checkpoint <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(readRDS(path),error=function(e)stop('Corrupted checkpoint: ',path,'; ',conditionMessage(e)))
}
check_phase_key <- function(phase) {
  key_file <- file.path(out,paste0(phase,'.run_key'))
  if (file.exists(key_file) && !identical(readLines(key_file,warn=FALSE),manifest$run_key))
    stop('Existing ',phase,' checkpoint is from a different run')
  key_file
}

pool_file <- file.path(out,'prefilter_pool.rds')
pool_key <- check_phase_key('prefilter')
if (file.exists(pool_file)) {
  if (!file.exists(pool_key)) stop('Prefilter checkpoint has no matching run key; refuse unsafe reuse')
  pool <- load_checkpoint(pool_file)
  cat('PHASE_PREFILTER=RESTORED\n')
} else {
  cat('PHASE_PREFILTER=START\n'); flush.console()
  # Independent phase seed makes resume from a completed prefilter checkpoint consistent.
  set.seed(as.integer(manifest$model$seed) + 101L)
  pool <- StableMate::lasso_prefilt(Y=ytrain,X=tr$X,env=env,
    p=as.integer(manifest$model$prefilter_p),K=as.integer(manifest$model$prefilter_K),
    ncore=as.integer(manifest$model$ncore),par_method='MC',family='binomial',
    group_by='Yenv',verbose=FALSE)
  if (length(pool)==0L) stop('Official prefilter returned empty predictor pool')
  writeLines(manifest$run_key,pool_key)
  save_atomic_rds(pool,pool_file)
  cat('PHASE_PREFILTER=SAVED\n')
}
writeLines(c(paste('class:',paste(class(pool),collapse=',')),
             paste('dimensions:',paste(dim(pool),collapse=' x ')),
             capture.output(str(pool,max.level=1,give.attr=FALSE))),
           file.path(out,'prefilter_pool_structure.txt'))

fit_file <- file.path(out,'official_model.rds')
fit_key <- check_phase_key('model')
if (file.exists(fit_file)) {
  if (!file.exists(fit_key)) stop('Model checkpoint has no matching run key; refuse unsafe reuse')
  model <- load_checkpoint(fit_file)
  cat('PHASE_OFFICIAL_MODEL=RESTORED\n')
} else {
  cat('PHASE_OFFICIAL_MODEL=START\n');flush.console()
  # Do not depend on RNG consumed by the (possibly restored) prefilter phase.
  set.seed(as.integer(manifest$model$seed) + 202L)
  model <- StableMate::stablemate(Y=ytrain,X=tr$X,env=env,
    K=as.integer(manifest$model$model_K),
    ncore=as.integer(manifest$model$ncore),par_method='MC',
    pred_st2_ctrl=list(reg_fun=get('reg_logit',ns),obj_fun=get('obj_bic_logit',ns),pred_pool=pool),
    stab_st2_ctrl=list(reg_fun=get('reg_logit',ns),obj_fun=get('obj_pss_logit',ns)),
    pred_calc_imp_ctrl=list(B=as.integer(manifest$model$bootstrap_B)),
    verbose=FALSE)
  if (is.null(model) || !length(model)) stop('Official model returned empty result')
  writeLines(manifest$run_key,fit_key)
  save_atomic_rds(model,fit_file)
  cat('PHASE_OFFICIAL_MODEL=SAVED\n')
}

# Author model prediction; do not silently substitute a new classifier.
prob <- as.numeric(stats::predict(model,X=te$X,pred_fun=get('pred_logit',ns)))
if (length(prob)!=length(ytest) || any(!is.finite(prob)) || any(prob<0 | prob>1))
  stop('Official model did not return one valid probability per test patient')
pred <- data.frame(outer=manifest$heldout,method='stablemate_author_native',
  ID=as.character(mtest$ID),Study=as.character(mtest$Study),
  class_true=as.character(mtest$class),y_true=ytest,
  probability_CD=prob,y_pred=as.integer(prob>=0.5))
write.csv(pred,file.path(out,'native_predictions.csv'),row.names=FALSE)

# Preserve native scores and selection definitions. A zero-selected result is valid.
if (is.null(model$prediction_ensemble$imp_scores$joint) ||
    is.null(model$stable_ensemble$imp_scores$conditional))
  stop('Author output shape changed; original RDS remains saved for forensic review')
ps <- model$prediction_ensemble$imp_scores$joint
ss <- model$stable_ensemble$imp_scores$conditional
save_atomic_rds(list(prediction=ps,stability=ss),file.path(out,'official_importance_objects.rds'))
if (is.null(ps$importance) || is.null(ps$significance) ||
    is.null(ss$importance) || is.null(ss$significance))
  stop('Official score object missing importance/significance; no invented scores')
genes <- intersect(names(ps$importance),names(ss$importance))
genes <- setdiff(genes,'pseudo_predictor')
if (!length(genes) || anyDuplicated(genes)) stop('No correctly named author gene scores')
scores <- data.frame(gene=genes,
    predictive_importance=as.numeric(ps$importance[genes]),
    stability_importance=as.numeric(ss$importance[genes]),
    predictive_significance=as.numeric(ps$significance[genes]),
    stability_significance=as.numeric(ss$significance[genes]))
scores$significantly_predictive <- !is.na(scores$predictive_significance) &
  scores$predictive_significance > 0.975
scores$significantly_stable_and_predictive <- scores$significantly_predictive &
  !is.na(scores$stability_significance) & scores$stability_significance > 0.975
scores <- scores[order(-scores$predictive_importance,scores$gene),]
write.csv(scores,file.path(out,'official_gene_scores.csv'),row.names=FALSE)
selected <- scores[scores$significantly_stable_and_predictive,,drop=FALSE]
write.csv(selected,file.path(out,'significant_stable_genes.csv'),row.names=FALSE)
write.csv(head(scores,50L),file.path(out,'diagnostic_top50_predictivity.csv'),row.names=FALSE)
cat('OFFICIAL_SCORED_GENES=',nrow(scores),'\n',sep='')
cat('OFFICIAL_SIGNIFICANT_PREDICTIVE_GENES=',sum(scores$significantly_predictive),'\n',sep='')
cat('OFFICIAL_SIGNIFICANT_STABLE_GENES=',nrow(selected),'\n',sep='')
cat('NATIVE_HELDOUT_PREDICTIONS=',nrow(pred),'\n',sep='')
cat('R_SESSION_BEGIN\n');print(sessionInfo());cat('R_SESSION_END\n')
cat('STAGE8_R_COMPLETE=TRUE\n')
