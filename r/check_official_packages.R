#!/usr/bin/env Rscript
# Diagnostic only: never installs packages or changes an R library.
# Installing official R methods MUST follow source/version/commit verification.
cat('R_VERSION=', as.character(getRversion()), '\n', sep='')
cat('R_PLATFORM=', R.version$platform, '\n', sep='')

check_pkg <- function(pkg, required_fns) {
  if (!requireNamespace(pkg, quietly=TRUE)) {
    cat(pkg, '_STATUS=NOT_INSTALLED\n', sep='')
    return(invisible(FALSE))
  }
  version <- as.character(utils::packageVersion(pkg))
  exports <- getNamespaceExports(pkg)
  missing <- setdiff(required_fns, exports)
  cat(pkg, '_VERSION=', version, '\n', sep='')
  cat(pkg, '_STATUS=', if (length(missing)) 'MISSING_EXPORTS' else 'EXPORTS_PRESENT', '\n', sep='')
  if (length(missing)) cat(pkg, '_MISSING=', paste(missing, collapse=','), '\n', sep='')
  invisible(length(missing)==0)
}

stablemate_ok <- check_pkg('StableMate', c('stablemate', 'lasso_prefilt'))
tentacles_ok <- check_pkg('TENTACLES', c('preProcess', 'runClassifiers', 'getConsensus', 'testConsensus', 'valConsensus'))
cat('TENTACLES_R_VERSION_OK=', if (getRversion() >= '4.5.0') 'YES' else 'NO', '\n', sep='')
cat('R_METHOD_SMOKE_TESTED=NO\n')
# Package export presence is not an algorithmic reproduction test.
