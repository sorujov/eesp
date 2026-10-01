# Round 9 (descriptive): robust linear grouping, RLGA (Garcia-Escudero et al. 2010, package lga
# 1.1-1 from the CRAN archive), at retained fractions 0.75 and 0.60 (trimming 0.25 and 0.40), on
# units [UNIT_START, UNIT_END) with stride UNIT_STRIDE.  RLGA fits orthogonal hyperplanes in the
# (covariates, response) space on standardised data; each hyperplane a'(v / s) = c is written
# back as a regression y = b0 + b'x.  Same output format as r_methods.R.
suppressMessages(library(lga))
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]; out_file <- args[2]; units_file <- args[3]
start <- as.integer(Sys.getenv("UNIT_START", "0"))
end <- as.integer(Sys.getenv("UNIT_END", "-1"))
stride <- as.integer(Sys.getenv("UNIT_STRIDE", "1"))
units <- read.csv(units_file, stringsAsFactors = FALSE)
if (end < 0) end <- nrow(units)
rows <- list()
for (i in seq(start + 1, end, by = stride)) {
  u <- units[i, ]
  dat <- read.csv(file.path(data_dir, paste0(u$name, ".csv")))
  xs <- grep("^x", names(dat), value = TRUE)
  V <- as.matrix(dat[, c(xs, "y")]); K <- u$K; d <- length(xs) + 1
  s <- sqrt(apply(V, 2, var))
  for (keep in c(0.75, 0.60)) {
    meth <- sprintf("RLGA(%.2f)", 1 - keep)
    set.seed(u$rseed + 17)
    t0 <- proc.time()[3]
    r <- tryCatch(suppressMessages(capture.output(res <- rlga(V, K, alpha = keep, silent = TRUE))), error = function(e) NULL)
    tt <- proc.time()[3] - t0
    b <- rep(NA_real_, 12); ah <- NA_real_
    if (!is.null(r) && exists("res") && !is.null(res$hpcoef)) {
      H <- matrix(res$hpcoef, nrow = K)
      bb <- c()
      for (k in 1:K) {
        a <- H[k, 1:(d)] / s; cc <- H[k, d + 1]      # a' v = cc in original units
        ay <- a[d]; ax <- a[-d]
        bb <- c(bb, cc / ay, -ax / ay)
      }
      b[1:length(bb)] <- bb
      ah <- mean(res$cluster == 0)
    }
    if (exists("res")) rm(res)
    rows[[length(rows) + 1]] <- c(list(unit = u$name, method = meth, time = tt, alpha_hat = ah),
                                  setNames(as.list(b), paste0("b", 1:12)))
  }
}
df <- do.call(rbind, lapply(rows, function(r) as.data.frame(r, stringsAsFactors = FALSE)))
write.csv(df, out_file, row.names = FALSE)
cat(sprintf("UNITS_DONE=%d\n", length(seq(start + 1, end, by = stride))))
