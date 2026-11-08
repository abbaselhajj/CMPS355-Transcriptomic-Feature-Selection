args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) stop("Expected input ZIP, output directory, expected Git SHA, and expected dataset SHA")
zip_path <- normalizePath(args[[1L]], mustWork = TRUE)
out_dir <- normalizePath(args[[2L]], mustWork = TRUE)
expected_source <- tolower(args[[3L]])
expected_data <- tolower(args[[4L]])
if (!grepl("^[a-f0-9]{40}$", expected_source)) stop("Invalid expected source revision")
if (!grepl("^[a-f0-9]{64}$", expected_data)) stop("Invalid expected data checksum")

library_dir <- Sys.getenv("CMPS355_R_LIB", "")
if (nzchar(library_dir) && dir.exists(library_dir)) .libPaths(c(library_dir, .libPaths()))
if (!requireNamespace("TENTACLES", quietly = TRUE)) stop("TENTACLES package not installed in the current R library")
package_info <- utils::packageDescription("TENTACLES")
source_sha <- tolower(package_info[["RemoteSha"]])
if (length(source_sha) != 1L || is.na(source_sha) || source_sha != expected_source) {
  stop("Installed TENTACLES revision does not match the pinned source")
}

cat("METHOD=TENTACLES\n")
cat("SCOPE=D1_ONLY_PUBLISHED_DISCOVERY_PREPROCESSING\n")
cat("SOURCE_SHA=", source_sha, "\n", sep = "")
cat("PACKAGE_VERSION=", as.character(utils::packageVersion("TENTACLES")), "\n", sep = "")
cat("R_VERSION=", as.character(getRversion()), "\n", sep = "")
cat("EXPECTED_DATA_SHA256=", expected_data, "\n", sep = "")

members <- utils::unzip(zip_path, list = TRUE)$Name
needed <- c("count_d1_aligned.csv", "desc_d1.csv")
if (!all(needed %in% members)) stop("Dataset ZIP does not contain both D1 input files")
counts <- utils::read.csv(unz(zip_path, needed[[1L]]), check.names = FALSE, stringsAsFactors = FALSE)
clinical <- utils::read.csv(unz(zip_path, needed[[2L]]), check.names = FALSE, stringsAsFactors = FALSE)
if (!("ID" %in% names(counts) && "ID" %in% names(clinical) && "class" %in% names(clinical))) {
  stop("Missing ID or clinical class column")
}
if (anyNA(counts$ID) || anyNA(clinical$ID) || anyDuplicated(counts$ID) || anyDuplicated(clinical$ID)) {
  stop("Missing or duplicate D1 sample IDs")
}
if (!identical(as.character(counts$ID), as.character(clinical$ID))) stop("D1 count and metadata sample IDs do not align")
row.names(counts) <- counts$ID
counts$ID <- NULL
row.names(clinical) <- clinical$ID
clinical <- data.frame(class = clinical$class, row.names = row.names(clinical), stringsAsFactors = FALSE)
if (anyDuplicated(names(counts)) || ncol(counts) < 9000L || nrow(counts) != 260L) stop("Unexpected D1 count matrix structure")
if (!identical(sort(unique(as.character(clinical$class))), c("CD", "Not IBD"))) stop("Unexpected D1 clinical labels")
if (!all(vapply(counts, is.numeric, logical(1L)))) stop("Non-numeric D1 gene column")
if (anyNA(counts) || any(!is.finite(as.matrix(counts))) || any(as.matrix(counts) < 0)) stop("Invalid D1 count values")
if (any(rowSums(as.matrix(counts)) <= 0)) stop("D1 contains a sample with a zero library size")

cat("INPUT_SAMPLES=", nrow(counts), "\n", sep = "")
cat("INPUT_GENES=", ncol(counts), "\n", sep = "")
cat("CLASS_CD=", sum(clinical$class == "CD"), "\n", sep = "")
cat("CLASS_NOT_IBD=", sum(clinical$class == "Not IBD"), "\n", sep = "")
cat("SAMPLE_ALIGNMENT=PASS\n")
cat("PREPROCESS_PARAMETERS=data.type:rnaseq;is.normalized:FALSE;mincpm:1;minfraction:0.1;batch:NULL;covar.mod:NULL;plot:FALSE\n")

result <- TENTACLES::preProcess(
  counts, clinical,
  class = "class", case.label = "CD",
  mincpm = 1, minfraction = 0.1,
  data.type = "rnaseq", is.normalized = FALSE,
  batch = NULL, covar.mod = NULL, plot = FALSE
)
if (is.null(result)) stop("Official preProcess returned NULL")

preprocessed_path <- file.path(out_dir, "d1_preprocessed.rds")
atomic_path <- paste0(preprocessed_path, ".partial")
saveRDS(result, atomic_path, compress = "gzip")
if (!file.rename(atomic_path, preprocessed_path)) stop("Failed to finalise preprocessed RDS")
cat("PREPROCESS_EXECUTED=TRUE\n")
cat("PREPROCESS_RESULT_CLASS=", paste(class(result), collapse = ","), "\n", sep = "")
cat("PREPROCESS_RESULT_NAMES=", paste(names(result), collapse = ","), "\n", sep = "")
cat("PREPROCESS_RDS_BYTES=", file.info(preprocessed_path)$size, "\n", sep = "")

api_lines <- c(
  paste0("Package: TENTACLES ", as.character(utils::packageVersion("TENTACLES"))),
  paste0("Source commit: ", source_sha),
  "Function: runClassifiers",
  capture.output(print(formals(TENTACLES::runClassifiers)))
)
writeLines(api_lines, file.path(out_dir, "runclassifiers_formals.txt"))
tryCatch({
  help_file <- utils:::.getHelpFile(utils::help("runClassifiers", package = "TENTACLES"))
  writeLines(capture.output(tools::Rd2txt(help_file)), file.path(out_dir, "runclassifiers_help.txt"))
  cat("AUTHOR_RUNCLASSIFIERS_DOCUMENTATION=AVAILABLE\n")
}, error = function(e) {
  cat("AUTHOR_RUNCLASSIFIERS_DOCUMENTATION=UNAVAILABLE\n")
  cat("DOCUMENTATION_ERROR=", conditionMessage(e), "\n", sep = "")
})

writeLines(capture.output(utils::sessionInfo()), file.path(out_dir, "r_session_info.txt"))
cat("RUNCLASSIFIERS_API_CAPTURED=TRUE\n")
cat("STATUS=PASS_D1_OFFICIAL_PREPROCESS\n")
