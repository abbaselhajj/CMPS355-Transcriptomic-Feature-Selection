#!/usr/bin/env Rscript
# Combine finished cross-cohort StableMate runs; builds paper-ready source tables
# and simple diagnostic figures. Never retrains or changes fold outputs.
options(stringsAsFactors=FALSE)
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L || !args[[1L]] %in% c('2','10','20','50','2000'))
  stop('Usage: Rscript --vanilla stablemate_stage8_collect.R 2000',call.=FALSE)
K <- as.integer(args[[1L]])
base <- file.path(getwd(),'runs','stablemate_continuous_final')
folds <- paste0('d',1:4)
read_csv <- function(path) {
  if(!file.exists(path)) stop('Missing: ',path,call.=FALSE)
  utils::read.csv(path,stringsAsFactors=FALSE,check.names=FALSE)
}
stopif <- function(cond,msg) if(!isTRUE(cond)) stop(msg,call.=FALSE)
metrics <- list();preds <- list();scores <- list();runs <- list()
for(fold in folds) {
  folder <- file.path(base,sprintf('%s_K%d_B1000',fold,K))
  stopif(file.exists(file.path(folder,'RUN_COMPLETE.txt')),
         paste('Fold not fully finished:',fold,'expected',folder))
  meta <- read_csv(file.path(folder,'run_summary.csv'))
  stopif(nrow(meta)==1L && identical(as.character(meta$heldout),fold) && meta$K[[1L]]==K,
         paste('Fold summary incompatible:',fold))
  # Verify saved evidence against the checksums written at run completion.
  sums <- read_csv(file.path(folder,'output_checksums.csv'))
  for(i in seq_len(nrow(sums))) {
    path <- file.path(folder,sums$file[[i]])
    stopif(file.exists(path),paste('Missing output:',path))
    got <- system2('sha256sum',shQuote(path),stdout=TRUE)
    stopif(length(got)==1L && substr(got,1L,64L)==sums$sha256[[i]],
           paste('Output checksum mismatch:',path))
  }
  m <- read_csv(file.path(folder,'heldout_metrics.csv'))
  p <- read_csv(file.path(folder,'heldout_predictions.csv'))
  s <- read_csv(file.path(folder,'author_gene_scores.csv'))
  stopif(nrow(m)==1L && all(p$heldout==fold) && all(s$heldout==fold),
         paste('Fold mismatch:',fold))
  stopif(nrow(p)==meta$test_n[[1L]] && length(unique(p$ID))==nrow(p),
         paste('Prediction count/ID error:',fold))
  metrics[[fold]] <- m;preds[[fold]] <- p;scores[[fold]] <- s;runs[[fold]] <- meta
}
metrics <- do.call(rbind,metrics);preds <- do.call(rbind,preds)
scores <- do.call(rbind,scores);runs <- do.call(rbind,runs)
row.names(metrics)<-NULL;row.names(preds)<-NULL;row.names(scores)<-NULL
stopif(!anyDuplicated(preds$ID),'Duplicate held-out patients across fold predictions')
stopif(all(preds$y_true %in% 0:1) && all(is.finite(preds$probability_CD)),
       'Invalid aggregated patient predictions')

# This summarizes independent held-out-cohort predictions from four different
# fitted models. Pooled AUC is descriptive, not a single-model external AUC.
metrics_fun <- function(y,p) {
  hat <- as.integer(p>=0.5);tp<-sum(y==1 & hat==1);tn<-sum(y==0 & hat==0)
  fp<-sum(y==0 & hat==1);fn<-sum(y==1 & hat==0)
  div<-function(x,z)if(z==0)NA_real_ else x/z
  auc<-(sum(rank(p,ties.method='average')[y==1])-sum(y==1)*(sum(y==1)+1)/2)/(sum(y==1)*sum(y==0))
  data.frame(n=length(y),tp=tp,tn=tn,fp=fp,fn=fn,
             roc_auc=auc,accuracy=mean(hat==y),sensitivity=div(tp,tp+fn),
             specificity=div(tn,tn+fp),f1=div(2*tp,2*tp+fp+fn),
             brier_score=mean((p-y)^2))
}
pooled <- metrics_fun(preds$y_true,preds$probability_CD)
pooled$interpretation <- 'Descriptive aggregation of four distinct outer-fold models'
metric_names <- c('roc_auc','accuracy','balanced_accuracy','sensitivity','specificity','precision','f1','brier_score','log_loss')
macro <- data.frame(metric=metric_names,
             mean_unweighted=vapply(metric_names,function(col)mean(metrics[[col]],na.rm=TRUE),numeric(1)),
             sd_unweighted=vapply(metric_names,function(col)stats::sd(metrics[[col]],na.rm=TRUE),numeric(1)))

# Reproducibility of feature selection: one held-out cohort per fold.
# Denominator tracks how many folds assigned a score to the gene; absence is
# not silently counted as a nonselection.
sp <- split(scores,scores$gene)
rec <- do.call(rbind,lapply(sp,function(d) data.frame(
  gene=d$gene[[1L]], folds_scored=nrow(d),
  folds_predictive=sum(d$significantly_predictive,na.rm=TRUE),
  folds_stable=sum(d$significantly_stable_predictive,na.rm=TRUE),
  mean_predictive_importance=mean(d$predictive_importance,na.rm=TRUE),
  mean_stability_importance=mean(d$stability_importance,na.rm=TRUE),
  heldout_folds_stable=paste(d$heldout[which(d$significantly_stable_predictive)],collapse=';'))))
rec <- rec[order(-rec$folds_stable,-rec$folds_scored,rec$gene),,drop=FALSE]
row.names(rec)<-NULL
out <- file.path(base,sprintf('aggregate_K%d',K))
if(file.exists(file.path(out,'AGGREGATE_COMPLETE.txt')))
  stop('Aggregate already completed. Refusing to overwrite: ',out,call.=FALSE)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(metrics,file.path(out,'table_fold_metrics.csv'),row.names=FALSE)
write.csv(macro,file.path(out,'table_macro_metrics.csv'),row.names=FALSE)
write.csv(pooled,file.path(out,'table_descriptive_pooled_metrics.csv'),row.names=FALSE)
write.csv(runs,file.path(out,'table_run_provenance.csv'),row.names=FALSE)
write.csv(preds,file.path(out,'all_heldout_predictions.csv'),row.names=FALSE)
write.csv(scores,file.path(out,'all_fold_gene_scores.csv'),row.names=FALSE)
write.csv(rec,file.path(out,'table_gene_reproducibility.csv'),row.names=FALSE)

png(file.path(out,'figure_heldout_roc_auc.png'),width=1300,height=850,res=140)
barplot(metrics$roc_auc,names.arg=metrics$heldout,ylim=c(0,1),
        xlab='Held-out cohort',ylab='ROC AUC',
        main='StableMate: leave-one-cohort-out discrimination')
abline(h=0.5,lty=2)
dev.off()

stable_count <- vapply(folds,function(f)sum(scores$heldout==f &
    scores$significantly_stable_predictive,na.rm=TRUE),numeric(1))
png(file.path(out,'figure_stable_gene_counts.png'),width=1300,height=850,res=140)
barplot(stable_count,names.arg=folds,
        xlab='Held-out cohort',ylab='Number of significant stable-and-predictive genes',
        main='StableMate: per-fold stable gene selection')
dev.off()

writeLines(c('Complete outputs from four separate leave-one-cohort-out models.',
 'Do not interpret pooled ROC-AUC as a single common model evaluation.',
 'Macro metrics are unweighted across the four independent held-out folds.',
 'Significant: author joint predictivity > 0.975 AND conditional stability > 0.975.',
 'A gene missing from a fold score table is not assigned an invented zero score.',
 'The full native bootstrap importance matrices remain inside each model_finished.rds.'),
 file.path(out,'README_INTERPRETATION.txt'))
writeLines(c('AGGREGATE_COMPLETE=TRUE',paste0('K=',K),
             paste0('TIMESTAMP=',format(Sys.time(),tz='UTC',usetz=TRUE))),
           file.path(out,'AGGREGATE_COMPLETE.txt'))
cat('AGGREGATE_COMPLETE path=',out,' patients=',nrow(preds),
    ' scored_gene_rows=',nrow(scores),'\n',sep='')
