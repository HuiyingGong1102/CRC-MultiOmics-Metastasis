# Check or install the R packages required by the demonstration.
required <- c(glmnet = "4.1-9", orthopolynom = "1.0-6.1", jsonlite = "2.0.0", nnls = "1.6")
install <- "--install" %in% commandArgs(trailingOnly = TRUE)
missing <- names(required)[!vapply(names(required), requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) && install) install.packages(missing, repos = "https://cloud.r-project.org")
missing <- names(required)[!vapply(names(required), requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "), ". Run with --install.")
for (p in names(required)) cat(p, as.character(packageVersion(p)), "(tested:", required[[p]], ")\n")
