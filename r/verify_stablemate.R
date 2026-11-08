
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected mode and commit")
mode <- args[[1L]]
commit <- args[[2L]]
if (!grepl("^[a-f0-9]{40}$", commit)) stop("A full Git commit is required")
if (!mode %in% c("install", "smoke")) stop("Unknown verification mode")

lib <- Sys.getenv("CMPS355_R_LIB", unset = "")
if (identical(lib, "")) stop("CMPS355_R_LIB is required")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
lib <- normalizePath(lib, mustWork = TRUE)
Sys.setenv(R_LIBS = lib, R_LIBS_USER = lib)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (identical(mode, "install")) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", lib = lib, quiet = TRUE)
  }
  remotes::install_github(
    "meiosis97/StableMate",
    ref = commit,
    subdir = "pkg",
    lib = lib,
    dependencies = NA,
    upgrade = "never",
    force = TRUE,
    build_vignettes = FALSE,
    quiet = FALSE
  )
}

if (!requireNamespace("StableMate", quietly = TRUE)) stop("StableMate is not installed")
metadata <- utils::packageDescription("StableMate")
sha <- metadata[["RemoteSha"]]
if (is.null(sha) || is.na(sha) || !identical(tolower(sha), tolower(commit))) {
  stop(paste("Installed StableMate revision does not match the requested commit:", sha))
}
exports <- getNamespaceExports("StableMate")
if (!all(c("stablemate", "lasso_prefilt") %in% exports)) {
  stop("Required author functions are missing")
}
cat("METHOD=StableMate\n")
cat("SOURCE_SHA=", sha, "\n", sep = "")
cat("VERSION=", as.character(utils::packageVersion("StableMate")), "\n", sep = "")
cat("R_VERSION=", as.character(getRversion()), "\n", sep = "")
cat("PACKAGE_LIBRARY=", lib, "\n", sep = "")

check_parallel_worker <- function() {
  if (!requireNamespace("doParallel", quietly = TRUE)) stop("doParallel is unavailable in the parent R process")
  if (!requireNamespace("foreach", quietly = TRUE)) stop("foreach is unavailable in the parent R process")
  cl <- parallel::makePSOCKcluster(1L)
  on.exit({
    foreach::registerDoSEQ()
    parallel::stopCluster(cl)
  }, add = TRUE)
  worker <- parallel::clusterEvalQ(cl, list(
    libraries = .libPaths(),
    doParallel = requireNamespace("doParallel", quietly = TRUE),
    foreach = requireNamespace("foreach", quietly = TRUE),
    StableMate = requireNamespace("StableMate", quietly = TRUE)
  ))[[1L]]
  cat("WORKER_LIBRARIES=", paste(worker$libraries, collapse = ";"), "\n", sep = "")
  for (name in c("doParallel", "foreach", "StableMate")) {
    cat("WORKER_PACKAGE_", name, "=", worker[[name]], "\n", sep = "")
  }
  if (!all(unlist(worker[c("doParallel", "foreach", "StableMate")], use.names = FALSE))) {
    stop("Parallel worker cannot load required packages; check WORKER_LIBRARIES")
  }
  doParallel::registerDoParallel(cl)
  `%dopar%` <- foreach::`%dopar%`
  values <- foreach::foreach(i = 1:2, .combine = c) %dopar% (i * i)
  if (!identical(as.integer(values), c(1L, 4L))) stop("Parallel foreach preflight returned unexpected values")
  cat("PARALLEL_PREFLIGHT=PASS\n")
}

if (identical(mode, "smoke")) {
  check_parallel_worker()
  set.seed(2026)
  n <- 100L
  X <- matrix(stats::rnorm(n * 6L), nrow = n, ncol = 6L)
  colnames(X) <- paste0("feature_", seq_len(ncol(X)))
  env <- rep(c("env_a", "env_b"), each = n / 2L)
  Y <- as.vector(1.5 * X[, 1L] - 0.7 * X[, 2L] + stats::rnorm(n, sd = 0.7))
  cat("EXECUTION_BACKEND=PSOCK_1_WORKER\n")
  fit <- StableMate::stablemate(Y = Y, X = X, env = env, K = 30L, ncore = 1L, verbose = FALSE)
  if (is.null(fit) || length(fit) == 0L) stop("StableMate returned an empty result")
  cat("SMOKE_FIT=PASS\n")
  cat("FIT_CLASS=", paste(class(fit), collapse = ","), "\n", sep = "")
}

cat("SESSION_INFO_BEGIN\n")
print(utils::sessionInfo())
cat("SESSION_INFO_END\n")