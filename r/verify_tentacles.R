args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected mode and source commit")
mode <- args[[1L]]
commit <- tolower(args[[2L]])
if (!mode %in% c("install", "probe", "smoke")) stop("Invalid mode")
if (!grepl("^[a-f0-9]{40}$", commit)) stop("Expected a full Git commit SHA")

lib <- Sys.getenv("CMPS355_R_LIB", unset = "")
if (identical(lib, "")) stop("CMPS355_R_LIB must be configured")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
lib <- normalizePath(lib, mustWork = TRUE)
Sys.setenv(R_LIBS = lib, R_LIBS_USER = lib)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (getRversion() < package_version("4.5.0")) stop("TENTACLES requires R >= 4.5.0")

cat("METHOD=TENTACLES\n")
cat("MODE=", mode, "\n", sep = "")
cat("REQUESTED_SHA=", commit, "\n", sep = "")
cat("R_VERSION=", as.character(getRversion()), "\n", sep = "")
cat("PACKAGE_LIBRARY=", lib, "\n", sep = "")

if (identical(mode, "install")) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", lib = lib)
  }
  source_dir <- Sys.getenv("CMPS355_TENTACLES_SOURCE", unset = "")
  desc <- file.path(source_dir, "DESCRIPTION")
  if (!file.exists(desc)) stop("Pinned TENTACLES source DESCRIPTION is missing")
  dependency_fields <- read.dcf(desc)[1L, , drop = TRUE]
  requirements <- paste(dependency_fields[intersect(names(dependency_fields), c("Depends", "Imports", "LinkingTo"))], collapse = ",")
  requirements <- trimws(gsub("\\s*\\([^)]*\\)", "", unlist(strsplit(requirements, ","))))
  bio_names <- c("edgeR", "sva", "pvca", "mixOmics", "limma", "Biobase", "SummarizedExperiment", "DESeq2", "org.Hs.eg.db", "AnnotationDbi", "ComplexHeatmap")
  missing_bio <- intersect(bio_names, requirements)
  missing_bio <- missing_bio[!vapply(missing_bio, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_bio)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager", lib = lib)
    }
    cat("BIOCONDUCTOR_DEPENDENCIES=", paste(missing_bio, collapse = ","), "\n", sep = "")
    BiocManager::install(missing_bio, ask = FALSE, update = FALSE, lib = lib)
  }
  remotes::install_github(
    "Giomu/TENTACLES",
    ref = commit,
    lib = lib,
    dependencies = NA,
    upgrade = "never",
    force = TRUE,
    build_vignettes = FALSE,
    quiet = FALSE
  )
}

if (!requireNamespace("TENTACLES", quietly = TRUE)) stop("Official TENTACLES package could not be loaded")
meta <- utils::packageDescription("TENTACLES")
installed_sha <- meta[["RemoteSha"]]
if (is.null(installed_sha) || is.na(installed_sha) || tolower(installed_sha) != commit) {
  stop(paste("Installed package SHA mismatch:", installed_sha))
}
cat("SOURCE_SHA=", installed_sha, "\n", sep = "")
cat("VERSION=", as.character(utils::packageVersion("TENTACLES")), "\n", sep = "")
exports <- getNamespaceExports("TENTACLES")
expected <- c("preProcess", "runClassifiers", "getConsensus", "testConsensus", "valConsensus")
for (function_name in expected) {
  cat("API_", function_name, "=", function_name %in% exports, "\n", sep = "")
}
if (!all(expected %in% exports)) stop("The five paper-defined functions are not all exported")
cat("AUTHOR_API=PASS\n")

if (identical(mode, "probe")) {
  for (name in expected) {
    cat("SIGNATURE_", name, "=", paste(deparse(args(getExportedValue("TENTACLES", name))), collapse = " "), "\n", sep = "")
  }
  data_entries <- utils::data(package = "TENTACLES")$results
  if (is.null(data_entries)) {
    cat("PACKAGE_DATASETS=none\n")
  } else {
    cat("PACKAGE_DATASETS=", paste(data_entries[, "Item"], collapse = ","), "\n", sep = "")
  }
  cat("PROBE=PASS\n")
}

if (identical(mode, "smoke")) {
  example_lines <- tryCatch(
    utils::example("preProcess", package = "TENTACLES", give.lines = TRUE),
    error = function(e) {
      cat("EXAMPLE_DISCOVERY_ERROR=", conditionMessage(e), "\n", sep = "")
      character()
    }
  )
  has_call <- length(example_lines) > 0L && any(grepl("\\bpreProcess\\s*\\(", example_lines))
  cat("AUTHOR_EXAMPLE_AVAILABLE=", has_call, "\n", sep = "")
  if (!has_call) {
    cat("AUTHOR_PREPROCESS_EXAMPLE=UNAVAILABLE\n")
  } else {
    set.seed(2026)
    work <- tempfile("tentacles_author_example_")
    dir.create(work)
    old <- setwd(work)
    on.exit(setwd(old), add = TRUE)
    utils::example("preProcess", package = "TENTACLES", ask = FALSE, echo = TRUE, run.dontrun = FALSE)
    cat("AUTHOR_PREPROCESS_EXAMPLE=PASS\n")
  }
}

cat("SESSION_INFO_BEGIN\n")
print(utils::sessionInfo())
cat("SESSION_INFO_END\n")
