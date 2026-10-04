#!/usr/bin/env Rscript
# Step 03: select minimum BIC and plot fitted protein values by module.
if (.Platform$OS.type == "windows") invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
root <- if (length(script)) dirname(dirname(normalizePath(script))) else getwd()
source(file.path(root, "R", "function.R"), encoding = "UTF-8")
suppressPackageStartupMessages(library(jsonlite))
out <- file.path(root, "results", "03_clustering_visualization")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
groups <- c("ColonT_NonMet", "ColonT_Liver", "ColonT_Lung", "ColonT_Other")
colors <- c("#39a432", "#7a52a6", "#e0271e", "#428dbf")
summary <- read.csv(file.path(root, "results", "02_clustering", "summary.csv"))
stopifnot(nrow(summary) == 90L, all(table(summary$cluster_number) == 10L),
          identical(sort(unique(summary$cluster_number)), 2:10), all(is.finite(summary$BIC)))
# Independently read each result JSON and check the summary score.
for (j in seq_len(nrow(summary))) {
  r <- summary[j, ]
  p <- file.path(root, "results", "02_clustering", sprintf("k_%02d", r$cluster_number),
                 sprintf("run_%02d", r$run), "result.json")
  result <- fromJSON(p)
  stopifnot(isTRUE(all.equal(result$BIC, r$BIC, tolerance = 1e-12)))
}
best <- do.call(rbind, lapply(split(summary, summary$cluster_number), function(x) {
  x <- x[order(x$BIC, x$run), ]
  x[1, , drop = FALSE]
}))
best <- best[order(best$cluster_number), ]
selected <- best[which.min(best$BIC), , drop = FALSE]
k <- selected$cluster_number
result_path <- file.path(root, "results", "02_clustering", sprintf("k_%02d", k),
                         sprintf("run_%02d", selected$run), "result.json")
result <- fromJSON(result_path)
data <- as.matrix(read.csv(file.path(root, "results", "01_read_data", "data_all.csv"),
                            row.names = 1, check.names = FALSE))
index <- as.matrix(read.csv(file.path(root, "results", "01_read_data", "index_all.csv"),
                             row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(root, "data", "demo_metadata.csv"))
stopifnot(nrow(index) == 1L, identical(colnames(data), colnames(index)),
          identical(rownames(data), result$protein_ids), !anyDuplicated(meta$Sample))
module <- as.integer(result$max_omega_logi) + 1L
stopifnot(length(module) == nrow(data), all(module %in% seq_len(k)),
          all(is.finite(data)), all(is.finite(index)))
sample_groups <- meta$Group[match(colnames(data), meta$Sample)]
stopifnot(!anyNA(sample_groups), all(sample_groups %in% groups))
counts <- tabulate(module, nbins = k)
stopifnot(sum(counts) == nrow(data))
save_plot <- function(stem, draw, width, height) {
  png(paste0(stem, ".png"), width = width, height = height, units = "in", res = 200)
  draw(); dev.off()
  tiff(paste0(stem, ".tiff"), width = width, height = height, units = "in", res = 300,
       compression = "lzw")
  draw(); dev.off()
}
plot_bic <- function() {
  par(mar = c(4.5, 5.5, 3.8, 1))
  plot(best$cluster_number, best$BIC, type = "o", pch = 19, col = "#cf3c3c", lwd = 2,
       xaxt = "n", xlab = "Number of modules (k)", ylab = "BIC", las = 1,
       main = "Minimum BIC across 10 runs per k")
  axis(1, at = best$cluster_number)
  abline(v = k, lty = 2, col = "#2864ac", lwd = 1.5)
  points(k, selected$BIC, pch = 21, cex = 1.6, bg = "#2864ac", col = "white")
  legend("topright", legend = sprintf("Minimum: k = %d, run = %d\nBIC = %.3f", k, selected$run, selected$BIC),
         bty = "n", text.col = "#2864ac", inset = c(0.18, 0))
}
save_plot(file.path(out, "BIC"), plot_bic, 7.5, 5)
# Same fit rule as the supplied code: OLS, with NNLS fallback for negative predictions.
fits <- vector("list", k)
for (m in seq_len(k)) {
  members <- which(module == m)
  fits[[m]] <- vector("list", length(groups))
  if (!length(members)) next
  for (g in seq_along(groups)) {
    columns <- which(sample_groups == groups[g])
    x <- as.numeric(index[1, columns])
    stopifnot(length(columns) >= 2L, all(diff(x) >= 0))
    y <- data[members, columns, drop = FALSE]
    protein_pars <- vapply(seq_len(nrow(y)), function(p) get_power_par(y[p, ], x), numeric(2))
    fitted <- vapply(seq_len(nrow(y)), function(p) lm_fit(protein_pars[, p], x), numeric(length(x)))
    mean_par <- get_power_par(y, x)
    mean_fit <- lm_fit(mean_par, x)
    stopifnot(all(is.finite(fitted)), all(is.finite(mean_fit)))
    fits[[m]][[g]] <- list(x = x, fitted = fitted, mean = mean_fit)
  }
}
plot_module <- function(m, labels = TRUE) {
  if (!counts[m]) {
    plot.new(); title(sprintf("M%d (0 proteins)", m)); return(invisible(NULL))
  }
  f <- fits[[m]]
  xr <- range(unlist(lapply(f, function(z) z$x)))
  yr <- range(unlist(lapply(f, function(z) c(z$fitted, z$mean))))
  pad <- max(diff(yr) * 0.06, 0.1)
  plot(NA, xlim = xr, ylim = yr + c(-pad, pad), xlab = if (labels) "Protein index (log2 total abundance)" else "",
       ylab = if (labels) "Fitted protein abundance (log2)" else "", las = 1,
       main = sprintf("M%d (%d proteins)", m, counts[m]), cex.main = 1)
  for (g in seq_along(groups)) {
    z <- f[[g]]
    points(rep(z$x, ncol(z$fitted)), as.vector(z$fitted), pch = 16, cex = 0.35,
           col = adjustcolor(colors[g], alpha.f = 0.07))
  }
  for (g in seq_along(groups)) lines(f[[g]]$x, f[[g]]$mean, col = colors[g], lwd = 2)
}
plot_overview <- function() {
  nc <- ceiling(sqrt(k)); nr <- ceiling(k / nc)
  par(mfrow = c(nr, nc), mar = c(2.5, 3.6, 2, 0.7), oma = c(3, 2, 3.5, 0), mgp = c(1.8, 0.5, 0))
  for (m in seq_len(k)) plot_module(m, labels = FALSE)
  mtext("Protein index (log2 total abundance)", side = 1, outer = TRUE, line = 1.2)
  mtext("Fitted protein abundance (log2)", side = 2, outer = TRUE, line = 0.4)
  mtext(sprintf("Selected model: k=%d, run=%d | faint points: protein fits; lines: module mean fits", k, selected$run),
        side = 3, outer = TRUE, line = 2, cex = 0.85)
  par(mfrow = c(1, 1), oma = rep(0, 4), fig = c(0, 1, 0, 1), mar = rep(0, 4), new = TRUE)
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i")
  legend(x = 0.5, y = 0.975, xjust = 0.5, yjust = 1, legend = groups, col = colors, lwd = 2, horiz = TRUE, bty = "n", cex = 0.85)
}
save_plot(file.path(out, "modules_overview"), plot_overview, 12, 10)
cat(sprintf("Selected k=%d, run=%d, BIC=%.6f\n", k, selected$run, selected$BIC))
cat("Module protein counts:", paste(counts, collapse = ", "), "\n")
cat("Verified 90 BIC scores, sample/protein order, finite fits, and module counts.\n")
