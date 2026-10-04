#!/usr/bin/env Rscript
# Run from any directory: Rscript /path/to/scripts/01_read_data.R
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
root <- if (length(script)) dirname(dirname(normalizePath(script))) else getwd()
if (.Platform$OS.type == "windows") invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
suppressPackageStartupMessages(library(parallel))
source(file.path(root, "R", "function.R"), encoding = "UTF-8")
options(cl.cores = 8L)
proteomics <- read.csv(file.path(root, "data", "demo_proteomics.csv"),
                       row.names = 1, check.names = FALSE)
metadata <- read.csv(file.path(root, "data", "demo_metadata.csv"),
                     check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(all(c("Sample", "Group") %in% names(metadata)),
          !anyDuplicated(metadata$Sample), !anyDuplicated(colnames(proteomics)),
          all(vapply(proteomics, is.numeric, logical(1))))
proteomics <- as.matrix(proteomics)
stopifnot(all(is.finite(proteomics)))
groups <- c("ColonT_NonMet", "ColonT_Liver", "ColonT_Lung", "ColonT_Other")
data_parts <- index_parts <- vector("list", length(groups))
sample_order <- character()
for (i in seq_along(groups)) {
  samples <- metadata$Sample[metadata$Group == groups[i]]
  stopifnot(length(samples) >= 2L, all(samples %in% colnames(proteomics)))
  abundance <- 2^proteomics[, samples, drop = FALSE] + 1
  stopifnot(all(is.finite(abundance)))
  totals <- colSums(abundance)
  index <- order(totals, method = "radix") # Stable ascending order for ties.
  abundance <- abundance[, index, drop = FALSE]
  totals <- totals[index]
  x <- log2(totals)
  y <- t(log2(abundance)) # equation_fit expects samples in rows.
  fit <- equation_fit(x = x, y = y, X_smooth = x, thread = 8)
  stopifnot(identical(dim(fit$original_data), dim(y)),
            identical(rownames(fit$original_data), samples[index]),
            isTRUE(all.equal(fit$original_data, y)),
            isTRUE(all.equal(fit$Time, x)), all(diff(fit$Time) >= 0),
            all(is.finite(fit$power_fit)))
  data_parts[[i]] <- t(fit$original_data)
  index_parts[[i]] <- matrix(as.numeric(fit$Time), nrow = 1L,
                           dimnames = list("Index", samples[index]))
  sample_order <- c(sample_order, samples[index])
  cat(sprintf("%s: %d proteins x %d samples; ascending total abundance verified\n",
              groups[i], nrow(abundance), ncol(abundance)))
}
data_all <- do.call(cbind, data_parts)
index_all <- do.call(cbind, index_parts)
stopifnot(identical(dim(data_all), c(nrow(proteomics), length(sample_order))),
          identical(dim(index_all), c(1L, length(sample_order))),
          identical(colnames(data_all), sample_order),
          identical(colnames(index_all), sample_order),
          identical(rownames(data_all), rownames(proteomics)),
          !anyDuplicated(sample_order))
# Independent check against the input matrix in the final sample order.
stopifnot(isTRUE(all.equal(data_all, log2(2^proteomics[, sample_order, drop = FALSE] + 1))),
          isTRUE(all.equal(as.numeric(index_all),
                           as.numeric(log2(colSums(2^proteomics[, sample_order, drop = FALSE] + 1))))))
dir.create(file.path(root, "results", "01_read_data"), showWarnings = FALSE, recursive = TRUE)
write.csv(data_all, file.path(root, "results", "01_read_data", "data_all.csv"), row.names = TRUE)
write.csv(index_all, file.path(root, "results", "01_read_data", "index_all.csv"), row.names = TRUE)
for (name in c("data_all", "index_all")) {
  expected <- get(name)
  actual <- as.matrix(read.csv(file.path(root, "results", "01_read_data", paste0(name, ".csv")),
                               row.names = 1, check.names = FALSE))
  stopifnot(identical(dim(actual), dim(expected)),
            identical(dimnames(actual), dimnames(expected)),
            isTRUE(all.equal(actual, expected, tolerance = 1e-12)))
  cat(sprintf("%s: %d x %d; sample order and CSV round-trip verified (tolerance 1e-12)\n",
              name, nrow(actual), ncol(actual)))
}
cat("Final sample order:\n", paste(sample_order, collapse = ","), "\n", sep = "")
