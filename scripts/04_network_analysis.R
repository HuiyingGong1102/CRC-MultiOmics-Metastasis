#!/usr/bin/env Rscript
# Step 04: coarse module networks and fine within-module protein networks.
if (.Platform$OS.type == "windows") invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
args <- commandArgs(FALSE)
f <- sub("^--file=", "", args[grepl("^--file=", args)])
root <- if (length(f)) dirname(dirname(normalizePath(f))) else getwd()
suppressPackageStartupMessages({library(parallel); library(glmnet); library(jsonlite); library(orthopolynom)})
source(file.path(root, "R", "function.R"), encoding = "UTF-8")
out <- file.path(root, "results", "04_network_analysis")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
base_seed <- 20261004L
summary <- read.csv(file.path(root, "results", "02_clustering", "summary.csv"))
selected <- summary[order(summary$BIC, summary$cluster_number, summary$run)[1], ]
model <- fromJSON(file.path(root, "results", "02_clustering", sprintf("k_%02d", selected$cluster_number),
                            sprintf("run_%02d", selected$run), "result.json"))
data <- as.matrix(read.csv(file.path(root, "results", "01_read_data", "data_all.csv"), row.names = 1, check.names = FALSE))
stopifnot(identical(rownames(data), model$protein_ids))
module <- as.integer(model$max_omega_logi) + 1L
k <- model$cluster_number
groups <- c("ColonT_NonMet", "ColonT_Liver", "ColonT_Lung", "ColonT_Other")
meta <- read.csv(file.path(root, "data", "demo_metadata.csv"))
sg <- meta$Group[match(colnames(data), meta$Sample)]
stopifnot(!anyNA(sg), length(module) == nrow(data))
cl <- makeCluster(8)
clusterEvalQ(cl, suppressPackageStartupMessages(library(glmnet)))
clusterExport(cl, c("get_interaction", "network_integral", "network_decompose", "lm_fit"))
records <- list()
run_network <- function(observed_linear, group, scale, module_name, seed) {
  folder <- file.path(out, scale, group, module_name)
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  totals <- rowSums(observed_linear)
  ix <- order(totals, method = "radix")
  time <- log2(totals[ix])
  y <- log2(observed_linear[ix, , drop = FALSE])
  stopifnot(all(is.finite(y)), length(unique(time)) > 1L, all(diff(time) >= 0))
  powers <- lapply(seq_len(ncol(y)), function(j) coef(lm(y[, j] ~ time)))
  names(powers) <- colnames(y)
  smooth <- seq(min(time), max(time), length.out = 100)
  interaction_x <- seq(min(time), max(time), length.out = 1000)
  clustered <- as.data.frame(vapply(powers, function(p) lm_fit(p, interaction_x), numeric(1000)), check.names = FALSE)
  # Use the supplied adaptive elastic-net selector, seeded per target.
  rel <- parLapply(cl, seq_len(ncol(y)), function(j, curves, s) {
    set.seed(s + j)
    if (ncol(curves) < 3L) return(list(colnames(curves)[j], character(), numeric()))
    r <- get_interaction(curves, j)
    stopifnot(length(r[[2]]) == length(r[[3]]))
    r
  }, curves = clustered, s = seed)
  decomposed <- parLapply(cl, rel, function(r, yy, tt, st, pp) network_decompose(yy, tt, st, pp, r),
                          yy = y, tt = time, st = smooth, pp = powers)
  edges <- list(); nodes <- list()
  for (j in seq_along(decomposed)) {
    d <- decomposed[[j]]
    stopifnot(all(is.finite(d$effects)), all(is.finite(d$total)),
              isTRUE(all.equal(d$total, rowSums(d$effects))))
    nodes[[j]] <- data.frame(Node = d$target, Module = module_name,
      Size = if (scale == "coarse_grained") sum(module == j) else 1L,
      SelfEffect = mean(d$effects[, 1]), FitObjective = d$objective, Convergence = d$convergence)
    if (length(d$dependencies)) for (h in seq_along(d$dependencies)) {
      effect <- mean(d$effects[, h + 1])
      edges[[length(edges) + 1L]] <- data.frame(From = d$dependencies[h], To = d$target,
        size = mean(d$effects[, 1]), Effect = effect, edge_type = if (effect > 0) 1L else 2L,
        weight = abs(effect))
    }
  }
  edges <- if (length(edges)) do.call(rbind, edges) else data.frame(From=character(), To=character(), size=numeric(), Effect=numeric(), edge_type=integer(), weight=numeric())
  nodes <- do.call(rbind, nodes)
  stopifnot(all(edges$From %in% nodes$Node), all(edges$To %in% nodes$Node), all(edges$From != edges$To))
  edges$size <- as.numeric(edges$size)^0.1 * 30
  edges$weight <- as.numeric(edges$weight)^0.1 * 2
  stopifnot(identical(names(edges), c("From", "To", "size", "Effect", "edge_type", "weight")),
            all(is.finite(edges$size)), all(is.finite(edges$weight)))
  links_path <- file.path(folder, paste0(group, "_links.csv"))
  write.csv(edges, links_path, row.names = FALSE, quote = FALSE)
  draw_page <- function(ids) {
    par(mfrow = c(3, 3), mar = c(3, 3.5, 2.2, 0.7), oma = c(1, 1, 3, 0), mgp = c(1.8, 0.5, 0))
    for (j in ids) {
      d <- decomposed[[j]]
      yr <- range(y[, j], d$effects, d$total)
      plot(time, y[, j], pch = 16, cex = 0.5, col = "#555555", ylim = yr + c(-1, 1) * max(diff(yr)*0.05, 0.05),
           xlab = "Index", ylab = "log2 abundance", main = d$target)
      abline(h = 0, lty = 3, col = "grey70")
      if (ncol(d$effects) > 1) matlines(smooth, d$effects[, -1, drop=FALSE], col = "#7DA449", lty = 1, lwd = 1)
      lines(smooth, d$effects[, 1], col = "#E72041", lwd = 1.5)
      lines(smooth, d$total, col = "#0099FF", lwd = 1.5)
    }
    mtext(paste(group, scale, module_name, "| black: observed; blue: total; red: self; green: incoming"), outer=TRUE, side=3, line=1, cex=0.8)
  }
  pdf(file.path(folder, "decomposition_curves.pdf"), width = 12, height = 10, useDingbats=FALSE)
  pages <- split(seq_len(ncol(y)), ceiling(seq_len(ncol(y)) / 9))
  for (ids in pages) draw_page(ids)
  dev.off()
  for (p in seq_along(pages)) {
    png(file.path(folder, sprintf("decomposition_%02d.png", p)), width=2400, height=2000, res=200)
    draw_page(pages[[p]]); dev.off()
  }
  back <- read.csv(links_path, check.names=FALSE)
  stopifnot(nrow(back) == nrow(edges), identical(back$From, edges$From), identical(back$To, edges$To),
            identical(names(back), names(edges)),
            isTRUE(all.equal(back$Effect, edges$Effect, tolerance=1e-12)),
            isTRUE(all.equal(back$size, edges$size, tolerance=1e-12)),
            isTRUE(all.equal(back$weight, edges$weight, tolerance=1e-12)))
  cat(sprintf("%s %s %s: %d nodes, %d edges, %d nonzero optimizer statuses\n", group, scale, module_name, ncol(y), nrow(edges), sum(nodes$Convergence != 0)))
  flush.console()
  data.frame(Group=group, Scale=scale, Module=module_name, Nodes=ncol(y), Edges=nrow(edges), Nonconverged=sum(nodes$Convergence != 0), Seed=seed)
}
tryCatch({
 for (g in seq_along(groups)) {
   # data_all already contains log2(2^original + 1); recover positive abundance.
   datt <- t(2^data[, sg == groups[g], drop = FALSE])
   coarse <- vapply(seq_len(k), function(m) rowMeans(datt[, module == m, drop = FALSE]), numeric(nrow(datt)))
   colnames(coarse) <- paste0("M", seq_len(k)); rownames(coarse) <- rownames(datt)
   records[[length(records)+1L]] <- run_network(coarse, groups[g], "coarse_grained", "all_modules", base_seed + g*10000L)
   for (mi in seq_len(k)) {
     # Fine network differs only in selecting all proteins within one module.
     data_observed <- datt[, which(module == mi), drop = FALSE]
     records[[length(records)+1L]] <- run_network(data_observed, groups[g], "fine_grained", paste0("M", mi), base_seed + g*10000L + mi*200L)
   }
 }
}, finally = stopCluster(cl))
stopifnot(length(records) == 4L * (k + 1L), sum(vapply(records, function(r) if(r$Scale == "fine_grained") r$Nodes else 0L, integer(1))) == 4L*nrow(data))
cat("Verified all coarse/fine networks, protein coverage, edge endpoints and CSV round trips.\n")
