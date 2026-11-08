#!/usr/bin/env Rscript
# Continuous StableMate LOCO: pinned author's algorithm, original native Lasso pool,
# exact completed-K logs for prediction/stability via the regression function hook,
# held-out predictions and reproducible final analysis exports.
# No intermediate model/predictor-pool checkpoints; no writes to original run folders.
# Usage: Rscript --vanilla stablemate_stage8_traced_final.R d1 1 10 [run|preflight|finalize]
options(stringsAsFactors = FALSE)

fail <- function(...) stop(..., call. = FALSE)
line <- function(...) { cat(sprintf(...), '\n', sep = ''); flush.console() }
sha256 <- function(path) {
  x <- suppressWarnings(system2('sha256sum', shQuote(path), stdout = TRUE, stderr = FALSE))
  if (length(x) != 1L || !grepl('^[0-9a-f]{64} ', x)) fail('SHA-256 failed for ',path)
  substr(x, 1L, 64L)
}
write_csv <- function(x, path) utils::write.csv(x, path, row.names = FALSE, na = '')
write_text <- function(text, path) writeLines(as.character(text), path, useBytes = TRUE)

binary_metrics <- function(y, p, heldout) {
  if (length(y) != length(p) || anyNA(y) || any(!y %in% 0:1) ||
      any(!is.finite(p)) || any(p < 0 | p > 1)) fail('Invalid inputs for binary metrics')
  hat <- as.integer(p >= 0.5)
  tp <- sum(y == 1L & hat == 1L); tn <- sum(y == 0L & hat == 0L)
  fp <- sum(y == 0L & hat == 1L); fn <- sum(y == 1L & hat == 0L)
  div <- function(a,b) if (b == 0) NA_real_ else a/b
  se <- div(tp,tp+fn); sp <- div(tn,tn+fp); prec <- div(tp,tp+fp)
  f1 <- div(2*tp,2*tp+fp+fn)
  auc <- if (length(unique(y)) == 2L) {
    (sum(rank(p, ties.method='average')[y == 1L]) - sum(y==1L)*(sum(y==1L)+1)/2) /
      (sum(y==1L)*sum(y==0L))
  } else NA_real_
  p_clip <- pmin(pmax(p,1e-15),1-1e-15)
  data.frame(heldout=heldout, n=length(y), n_negative=sum(y==0L), n_positive=sum(y==1L),
             tn=tn,fp=fp,fn=fn,tp=tp,threshold=0.5,
             roc_auc=auc,accuracy=mean(hat == y),balanced_accuracy=mean(c(se,sp)),
             sensitivity=se,specificity=sp,precision=prec,f1=f1,
             brier_score=mean((p-y)^2),log_loss=-mean(y*log(p_clip)+(1-y)*log(1-p_clip)))
}

main <- function() {
  args <- commandArgs(trailingOnly=TRUE)
  if (!(length(args) %in% c(3L,4L))) fail('Usage: Rscript --vanilla stablemate_stage8_traced_final.R d1 1 10 [run|preflight|finalize]')
  fold <- args[[1L]]; cores <- suppressWarnings(as.integer(args[[2L]]))
  K <- suppressWarnings(as.integer(args[[3L]]))
  mode <- if (length(args)==4L) args[[4L]] else 'run'
  if (!fold %in% paste0('d',1:4) || is.na(cores) || cores != 1L ||
      is.na(K) || !K %in% c(2L,10L,20L,50L,2000L) ||
      !mode %in% c('run','preflight','finalize'))
    fail('Allowed: d1..d4; exactly 1 worker; K 2/10/20/50/2000; run/preflight/finalize')
  root <- normalizePath(getwd(), mustWork=TRUE)
  lib <- file.path(root,'r_lib')
  if (!dir.exists(lib)) fail('Local R library not found: ',lib)
  .libPaths(unique(c(lib,.libPaths())))
  Sys.setenv(R_LIBS_USER=lib, R_LIBS=lib, OPENBLAS_NUM_THREADS='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
  if (!requireNamespace('StableMate',quietly=TRUE) || !requireNamespace('jsonlite',quietly=TRUE))
    fail('Required preinstalled packages StableMate/jsonlite unavailable')
  suppressPackageStartupMessages(library(Matrix))
  expected_sha <- 'cf0fe9f344c1756b6f89fd33f9fdae1a7f35b24a'
  installed_sha <- tolower(as.character(utils::packageDescription('StableMate')[['RemoteSha']]))
  if (!identical(installed_sha,expected_sha) ||
      as.character(utils::packageVersion('StableMate')) != '0.1.0') fail('StableMate author source/version changed')
  ns <- asNamespace('StableMate')
  auth <- function(nm) {
    if (!exists(nm, envir=ns, inherits=FALSE)) fail('Missing author function ', nm)
    get(nm,envir=ns)
  }
  invisible(lapply(c('reg_logit','obj_bic_logit','obj_pss_logit','pred_logit'),auth))

  official <- file.path(root,'runs','stablemate_k2000',fold)
  manifest_path <- file.path(official,'manifest.json')
  if (!file.exists(manifest_path)) fail('Missing prepared ',fold,' manifest: ',manifest_path)
  manifest <- jsonlite::read_json(manifest_path,simplifyVector=TRUE)
  if (!identical(manifest$heldout,fold) || !identical(manifest$source_commit,expected_sha) ||
      as.integer(manifest$model$model_K) != 2000L ||
      as.integer(manifest$model$prefilter_K) != 2000L ||
      as.integer(manifest$model$prefilter_p) != 50L ||
      as.integer(manifest$model$bootstrap_B) != 1000L)
    fail('Prepared manifest deviates from research protocol')
  B <- as.integer(manifest$model$bootstrap_B)
  seed <- as.integer(manifest$model$seed)
  if (length(seed)!=1L || is.na(seed)) fail('Missing experiment seed')
  work <- file.path(root,'work',fold)
  files <- c('train_x.csv','train_meta.csv','test_x.csv','test_meta.csv')
  checksums <- character(length(files)); names(checksums) <- files
  for (nm in files) {
    path <- file.path(work,nm)
    if (!file.exists(path)) fail('Missing ',fold,' prepared input: ',path)
    checksums[[nm]] <- sha256(path)
    expected <- manifest$input_files_sha256[[nm]]
    if (is.null(expected) || !identical(checksums[[nm]],as.character(expected)))
      fail('Prepared input SHA mismatch for ',fold,'/',nm)
  }
  read_x <- function(split) {
    df <- utils::read.csv(file.path(work,paste0(split,'_x.csv')),
                          check.names=FALSE,stringsAsFactors=FALSE)
    if (ncol(df)<2L || names(df)[1L]!='ID' || anyDuplicated(names(df)) || anyNA(df$ID))
      fail('Malformed ',split,'_x.csv')
    X <- as.matrix(df[,-1L,drop=FALSE]);storage.mode(X) <- 'double'
    if (any(!is.finite(X)) || anyDuplicated(colnames(X)) || ncol(X)<50L) fail('Bad numeric predictor matrix ',split)
    list(ids=as.character(df$ID), X=X)
  }
  read_meta <- function(split, ids) {
    df <- utils::read.csv(file.path(work,paste0(split,'_meta.csv')),
                          check.names=FALSE, stringsAsFactors=FALSE)
    if (!all(c('ID','Study','class','y','cohort') %in% names(df)) ||
        !identical(as.character(df$ID),ids) || anyNA(df[,c('ID','class','y','cohort')]))
      fail('Metadata malformed or out of row order in ',split)
    if (any(!df$y %in% c(0,1)) || any(as.integer(df$y)!=as.integer(df$class=='CD')))
      fail('Invalid label encoding in ',split)
    df
  }
  tr <- read_x('train'); te <- read_x('test')
  mtr <- read_meta('train',tr$ids); mte <- read_meta('test',te$ids)
  if (!identical(colnames(tr$X),colnames(te$X)) ||
      anyDuplicated(c(tr$ids,te$ids)) || length(unique(mtr$cohort)) != 3L ||
      !identical(sort(unique(as.character(mtr$cohort))),sort(as.character(manifest$training_cohorts))) ||
      any(mte$cohort != fold) || any(mte$cohort %in% mtr$cohort) ||
      nrow(tr$X) != manifest$train_patients || nrow(te$X) != manifest$test_patients ||
      ncol(tr$X) != manifest$eligible_genes || length(unique(mtr$y)) != 2L)
    fail('Train/heldout split integrity or manifest dimensions failed for ',fold)
  y <- as.integer(mtr$y); env <- as.character(mtr$cohort)
  X <- tr$X; X_test <- te$X
  out <- file.path(root,'runs','stablemate_continuous_final',
                   sprintf('%s_K%d_B%d',fold,K,B))
  line('PREFLIGHT_PASS fold=%s train=%d test=%d genes=%d K=%d B=%d cores=1',
       fold,nrow(X),nrow(X_test),ncol(X),K,B)
  if (mode == 'preflight') return(invisible(TRUE))

  model_path <- file.path(out,'model_finished.rds')
  if (mode == 'run' && (file.exists(model_path) || file.exists(file.path(out,'RUN_COMPLETE.txt'))))
    fail('Existing model/results; will not overwrite: ',out)
  if (mode == 'finalize' && file.exists(file.path(out,'RUN_COMPLETE.txt')))
    fail('Run is already completed; no finalization needed: ',out)
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  lock <- file.path(out,'RUNNING.lock')
  if (!dir.create(lock,showWarnings=FALSE)) fail('Fold already running or stale RUNNING.lock at: ',lock)
  on.exit(unlink(lock,recursive=TRUE),add=TRUE)
  # Carry the original, immutable preparation evidence into the result folder.
  evidence <- c('gene_transform.csv','outer_split.csv')
  for (nm in evidence) {
    src <- file.path(official,nm)
    expected_evidence <- manifest$audit_artifact_sha256[[nm]]
    if (is.null(expected_evidence) || !file.exists(src) ||
        !identical(sha256(src),as.character(expected_evidence)))
      fail('Prepared provenance artifact missing/mismatched: ',src)
    dest <- file.path(out,nm)
    if (!file.exists(dest) && !file.copy(src,dest)) fail('Could not copy ',nm)
    if (sha256(dest)!=as.character(expected_evidence)) fail('Copied provenance checksum mismatch: ',nm)
  }
  source_manifest <- file.path(out,'source_manifest.json')
  if (!file.exists(source_manifest) && !file.copy(manifest_path,source_manifest))
    fail('Cannot copy original manifest')
  if (sha256(source_manifest)!=sha256(manifest_path)) fail('Manifest copy mismatch')
  state_file <- file.path(out,'status.txt')
  write_text(c('STATUS=RUNNING',paste0('TIME=',format(Sys.time(),tz='UTC',usetz=TRUE))),state_file)
  done <- FALSE
  on.exit({
    if (!done) write_text(c('STATUS=FAILED_OR_INTERRUPTED',paste0('TIME=',format(Sys.time(),tz='UTC',usetz=TRUE))),state_file)
  },add=TRUE)
  progress_file <- file.path(out,'progress.csv')
  script_path <- file.path(root,'stablemate_stage8_traced_final.R')
  source_script_sha <- if(file.exists(script_path)) sha256(script_path) else NA_character_
  started <- Sys.time()
  phases <- data.frame(phase=character(),seconds=numeric())

  if (mode == 'run') {
    write_text('timestamp_utc,phase,completed_K,total_K,elapsed_seconds,eta_seconds',progress_file)
    line('SCRIPT_ID=stage8_traced_final_v1')
    line('RUN_START fold=%s K=%d B=%d ntrain=%d ntest=%d genes=%d',fold,K,B,nrow(X),nrow(X_test),ncol(X))
    line('AUTHOR_SHA=%s',installed_sha)
    line('OUTPUT=%s',out)
    line('PREFILTER_START fold=%s K=%d (phase-level logging; native author routine)',fold,K)
    set.seed(seed+101L)
    t0 <- proc.time()[['elapsed']]
    pool <- StableMate::lasso_prefilt(Y=y,X=X,env=env,
            p=as.integer(manifest$model$prefilter_p),K=K,ncore=1L,
            par_method='MC',family='binomial',group_by='Yenv',verbose=FALSE)
    pre_secs <- proc.time()[['elapsed']]-t0
    dense <- as.matrix(pool); cols <- colnames(dense)
    checks <- c(rows=nrow(dense)==K,names=!is.null(cols)&&!anyDuplicated(cols)&&
        all(cols %in% c('pseudo_predictor',colnames(X))),
        pseudo='pseudo_predictor' %in% cols, values=all(is.finite(dense))&&all(dense %in% 0:1),
        nonempty=all(rowSums(dense)>0))
    if (!all(checks)) fail('Native predictor pool invalid: ',paste(names(checks)[!checks],collapse=','))
    line('PREFILTER_PASS K=%d pool_width=%d elapsed_seconds=%.2f',K,ncol(pool),pre_secs)
    phases <- rbind(phases,data.frame(phase='prefilter',seconds=pre_secs))
    rm(dense)

    # The author's st2e() invokes reg_fun ONCE when each selected member is finished
    # (st2e.R: one_st2e_itr). With 1 MC worker, chunk_size=K: a single worker
    # handles all iterations per phase. The wrapper only records completed work;
    # the return value is exactly the original author's reg_logit() result.
    reg_orig <- auth('reg_logit')
    make_logged_reg <- function(phase) {
      n_done <- 0L
      t_first <- NULL
      function(Y, X, env, pred_in) {
        result <- reg_orig(Y, X, env, pred_in)
        n_done <<- n_done + 1L
        now <- Sys.time()
        if (is.null(t_first)) t_first <<- now
        elapsed <- as.numeric(difftime(now,t_first,units='secs'))
        eta <- if(n_done>1L) elapsed / (n_done-1L) * (K-n_done) else NA_real_
        record <- sprintf('%s,%s,%d,%d,%.2f,%.2f',
             format(now,'%Y-%m-%dT%H:%M:%SZ',tz='UTC'),phase,n_done,K,elapsed,eta)
        cat(record,'\n',file=progress_file,append=TRUE,sep='')
        if(n_done==1L || n_done%%25L==0L || n_done==K)
          cat(sprintf('PROGRESS fold=%s phase=%s completed=%d/%d elapsed_min=%.1f eta_min=%.1f\n',
              fold,phase,n_done,K,elapsed/60,eta/60))
        result
      }
    }
    line('ENSEMBLE_START fold=%s K=%d (progress.csv records EACH completed K)',fold,K)
    set.seed(seed+202L)
    t1 <- proc.time()[['elapsed']]
    model <- withCallingHandlers(StableMate::stablemate(Y=y,X=X,env=env,K=K,ncore=1L,par_method='MC',
       pred_st2_ctrl=list(reg_fun=make_logged_reg('prediction'),
            obj_fun=auth('obj_bic_logit'),pred_pool=pool),
       stab_st2_ctrl=list(reg_fun=make_logged_reg('stability'),
            obj_fun=auth('obj_pss_logit')),
       pred_calc_imp_ctrl=list(B=B,scale=TRUE),verbose=FALSE),
       message=function(m) {
         if (grepl('stable ensemble is built based',conditionMessage(m),fixed=TRUE))
           line('IMPORTANCE_START fold=%s B=%d (the two K searches have finished)',fold,B)
       })
    ens_secs <- proc.time()[['elapsed']]-t1
    if (!inherits(model,'stablemate') || !isTRUE(model$matched)) fail('Author returned invalid/unmatched StableMate model')
    for (part in c('prediction_ensemble','stable_ensemble')) {
      node <- model[[part]]
      if (is.null(node) || nrow(node$ensemble)!=K || length(node$obj_scores)!=K ||
          any(!is.finite(node$obj_scores))) fail('Invalid fitted ',part,' object/scores')
    }
    phases <- rbind(phases,data.frame(phase='ensemble_and_importance',seconds=ens_secs))
    line('ENSEMBLE_PASS fold=%s K=%d elapsed_seconds=%.2f',fold,K,ens_secs)

    # Save only a FINISHED model, never an intermediate pool or a resumable checkpoint.
    # If reporting fails, mode=finalize can regenerate exports from this finished model.
    saveRDS(model,model_path,compress=TRUE)
    line('FINISHED_MODEL_SAVED=%s',model_path)
    write_csv(phases,file.path(out,'phase_metrics.csv'))
  } else {
    if (!file.exists(model_path)) fail('No FINISHED model for finalize mode: ',model_path)
    model <- readRDS(model_path)
    if (!inherits(model,'stablemate') || !isTRUE(model$matched)) fail('Saved model invalid')
    for (part in c('prediction_ensemble','stable_ensemble'))
      if (nrow(model[[part]]$ensemble)!=K) fail('Saved K mismatch: ',part)
    line('FINALIZE_FROM_FINISHED_MODEL=%s',model_path)
  }

  # Final export. Training labels, held-out labels, and cohort membership stay separate.
  pred <- as.numeric(stats::predict(model,X=X_test,pred_fun=auth('pred_logit')))
  if (length(pred)!=nrow(X_test) || any(!is.finite(pred)) || any(pred<0|pred>1))
    fail('Native author predictor did not return valid held-out probabilities')
  pred_table <- data.frame(heldout=fold,method='stablemate_author_native',
            ID=as.character(mte$ID),Study=as.character(mte$Study),
            class_true=as.character(mte$class),y_true=as.integer(mte$y),
            probability_CD=pred,y_pred=as.integer(pred>=0.5))
  write_csv(pred_table,file.path(out,'heldout_predictions.csv'))
  metr <- binary_metrics(as.integer(mte$y),pred,fold)
  write_csv(metr,file.path(out,'heldout_metrics.csv'))

  ps <- model$prediction_ensemble$imp_scores$joint
  ss <- model$stable_ensemble$imp_scores$conditional
  for (nm in c('importance','significance','lt_significance'))
    if (is.null(ps[[nm]]) || is.null(ss[[nm]])) fail('Author importance result missing: ',nm)
  genes <- intersect(names(ps$importance),names(ss$importance))
  genes <- setdiff(genes,'pseudo_predictor')
  if (!length(genes) || anyDuplicated(genes)) fail('No correctly named author gene scores')
  lookup <- function(obj,field) as.numeric(obj[[field]][genes])
  if (anyNA(lookup(ps,'importance')) || anyNA(lookup(ss,'importance')))
    fail('Missing importance scores for aligned gene names')
  scores <- data.frame(heldout=fold,gene=genes,
        predictive_importance=lookup(ps,'importance'),
        stability_importance=lookup(ss,'importance'),
        predictive_significance=lookup(ps,'significance'),
        stability_significance=lookup(ss,'significance'),
        predictive_lt_significance=lookup(ps,'lt_significance'),
        stability_lt_significance=lookup(ss,'lt_significance'))
  scores$significantly_predictive <- !is.na(scores$predictive_significance) & scores$predictive_significance>0.975
  scores$significantly_stable_predictive <- scores$significantly_predictive &
        !is.na(scores$stability_significance) & scores$stability_significance>0.975
  freq <- function(which_ensemble) {
    a <- model[[which_ensemble]]$ensemble
    v <- as.numeric(colMeans(a))
    names(v) <- colnames(a)
    as.numeric(v[match(genes,names(v))])
  }
  scores$predictive_selection_frequency <- freq('prediction_ensemble')
  scores$stable_selection_frequency <- freq('stable_ensemble')
  if (anyNA(scores$predictive_selection_frequency) || anyNA(scores$stable_selection_frequency))
    fail('Failed to align per-gene ensemble selection frequencies')
  scores <- scores[order(-scores$predictive_importance,scores$gene),,drop=FALSE]
  write_csv(scores,file.path(out,'author_gene_scores.csv'))
  write_csv(scores[scores$significantly_stable_predictive,,drop=FALSE],
            file.path(out,'significant_stable_genes.csv'))
  write_csv(head(scores,50L),file.path(out,'diagnostic_top50_predictive.csv'))
  summary <- data.frame(heldout=fold,K=K,B=B,workers=1L,
          train_n=nrow(X),test_n=nrow(X_test),eligible_genes=ncol(X),
          scored_genes=nrow(scores),predictive_significant=sum(scores$significantly_predictive),
          stable_significant=sum(scores$significantly_stable_predictive),
          source_commit=installed_sha,manifest_run_key=manifest$run_key,
          script_sha256=source_script_sha)
  write_csv(summary,file.path(out,'run_summary.csv'))
  write_csv(as.data.frame(table(split=c(rep('train',nrow(mtr)),rep('heldout',nrow(mte))),
           cohort=c(as.character(mtr$cohort),as.character(mte$cohort)),
           y=c(as.integer(mtr$y),as.integer(mte$y)))),file.path(out,'split_counts.csv'))
  write_csv(data.frame(file=names(checksums),sha256=unname(checksums)),
            file.path(out,'input_checksums.csv'))
  write_text(c(sprintf('STABLEMATE_VERSION=%s',utils::packageVersion('StableMate')),
               paste0('STABLEMATE_COMMIT=',installed_sha),
               paste0('MANIFEST_RUN_KEY=',manifest$run_key),
               paste0('R_VERSION=',getRversion()),
               paste0('K=',K),paste0('B=',B),
               'PREFILTER_P=50','PREFILTER_GROUP_BY=Yenv',
               'PREFILTER_NATIVE_LASSO_OBJECT=TRUE',
               'REGRESSION=author reg_logit (logged completion hook delegates unchanged)',
               'OBJECTIVES=author obj_bic_logit + obj_pss_logit',
               'PREDICTION=author predict stablemate + pred_logit',
               'PREDICTIVE_IMPORTANCE=joint','STABILITY_IMPORTANCE=conditional',
               'AUTHOR_IMPORTANCE_SCALE=TRUE (explicit, as in successful benchmark)',
               'SIGNIFICANCE_RULE=both author significance > 0.975',
               'SEED_PREFILTER=manifest seed + 101','SEED_ENSEMBLE=manifest seed + 202',
               'EXECUTION=ncore 1, par_method MC, no checkpoints',
               'KNOWN_WARNING=binomial glm fractional outcome-environment weights',
               paste0('SCRIPT_SHA256=',source_script_sha)),file.path(out,'protocol.txt'))
  write_text(capture.output(sessionInfo()),file.path(out,'session_info.txt'))
  if (file.exists(progress_file) && mode=='run') {
    pp <- utils::read.csv(progress_file,stringsAsFactors=FALSE)
    valid <- all(c('phase','completed_K','total_K') %in% names(pp)) &&
       nrow(pp)==2L*K &&
       identical(as.integer(pp$completed_K[pp$phase=='prediction']),seq_len(K)) &&
       identical(as.integer(pp$completed_K[pp$phase=='stability']),seq_len(K))
    if (!valid) fail('Completed model preserved but progress trace failed validation. Inspect ',progress_file)
    line('PROGRESS_TRACE_PASS fold=%s prediction=%d/%d stability=%d/%d',fold,K,K,K,K)
  }
  # Final outputs checksum list for downstream provenance; include model and result tables.
  output_files <- c('model_finished.rds','heldout_predictions.csv','heldout_metrics.csv',
                    'author_gene_scores.csv','significant_stable_genes.csv','run_summary.csv',
                    'input_checksums.csv','protocol.txt','session_info.txt',
                    'source_manifest.json','gene_transform.csv','outer_split.csv')
  output_files <- c(output_files,intersect(c('progress.csv','phase_metrics.csv'),
                                          list.files(out)))
  write_csv(data.frame(file=output_files,
                       sha256=vapply(file.path(out,output_files),sha256,character(1))),
            file.path(out,'output_checksums.csv'))
  write_text(c('RUN_COMPLETE=TRUE',paste0('FOLD=',fold),paste0('K=',K),
               paste0('TIME_UTC=',format(Sys.time(),tz='UTC',usetz=TRUE))),
             file.path(out,'RUN_COMPLETE.txt'))
  done <- TRUE
  write_text(c('STATUS=COMPLETE',paste0('TIME=',format(Sys.time(),tz='UTC',usetz=TRUE))),state_file)
  line('RUN_COMPLETE fold=%s K=%d saved_predictions=%d scored_genes=%d',fold,K,nrow(pred_table),nrow(scores))
  line('OUTPUT=%s',out)
  invisible(TRUE)
}

main()
