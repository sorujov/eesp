# Laplace mixture of regressions (Song, Yao & Xing 2014), RobMixReg::mixLp with 20 starts.
# Added after the pre-registered study, descriptive only.  Same input/output format as r_methods.R.
suppressMessages(library(RobMixReg))
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]; out_file <- args[2]; units_file <- args[3]
start <- as.integer(Sys.getenv("UNIT_START", "0"))
end <- as.integer(Sys.getenv("UNIT_END", "-1"))
units <- read.csv(units_file, stringsAsFactors = FALSE)
nit <- as.integer(Sys.getenv("LP_NIT", "20"))
keep_re <- Sys.getenv("LP_DESIGNS", "")
if (end < 0) end <- nrow(units)
coefs_from <- function(cc, d, K) {
  cc <- as.matrix(cc)
  if (ncol(cc) != K || nrow(cc) < d) return(rep(NA_real_, K * d))
  as.vector(cc[1:d, , drop = FALSE])
}
rows <- list()
for (i in (start + 1):end) {
  u <- units[i, ]
  if (nchar(keep_re) > 0 && !grepl(keep_re, u$name)) next
  dat <- read.csv(file.path(data_dir, paste0(u$name, ".csv")))
  xs <- grep("^x", names(dat), value = TRUE)
  dd <- dat[, c(xs, "y")]
  K <- u$K; d <- length(xs) + 1
  X <- as.matrix(dat[, xs])
  run <- function(meth, expr, trimmed_fun) {
    set.seed(u$rseed + nchar(meth) * 7919)
    t0 <- proc.time()[3]
    r <- tryCatch(expr(), error = function(e) NULL)
    tt <- proc.time()[3] - t0
    b <- if (is.null(r)) rep(NA_real_, K * d) else coefs_from(r@compcoef, d, K)
    b <- c(b, rep(NA_real_, 12 - length(b)))
    ah <- if (is.null(r)) NA_real_ else trimmed_fun(r)
    rows[[length(rows) + 1]] <<- c(list(unit = u$name, method = meth, time = tt, alpha_hat = ah),
                                   setNames(as.list(b), paste0("b", seq_along(b))))
  }
  run(if (nit == 20) "Laplace" else paste0("Laplace", nit), function() mixLp(y ~ ., dd, nc = K, nit = nit), function(r) 0)
}
df <- do.call(rbind, lapply(rows, function(r) as.data.frame(r, stringsAsFactors = FALSE)))
write.csv(df, out_file, row.names = FALSE)
cat(sprintf("UNITS_DONE=%d\n", end - start))
