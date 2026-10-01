# R competitors for the v4 study (RobMixReg).  Reads the unit CSVs written by
# simv4.py and writes one CSV per chunk: unit, method, time, alpha_hat, b_k_j.
#   tcwm(a)  : trimmed cluster-weighted model, trim.cwm, trimming level a
#   TLE(a)   : trimmed likelihood estimator, TLE, trimming ratio a
#   CTLE     : component-wise adaptive trimming, CTLERob (no level)
#   bisq     : bisquare M-estimation for mixtures of regressions (Bai, Yao &
#              Boyer 2012), mixlinrb_bi; RobMixReg 1.1.3 fails with
#              "object 'w' not found" (lm() looks up the weights in the
#              formula's environment), so the package's own code is used with
#              the formula's environment set to the function frame; nothing else changed.
#   trim.cwm is run with 50 random starts and 20 concentration steps (the
#   package defaults are 20 and 10), to give it every chance.
suppressMessages(library(RobMixReg))

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]; out_file <- args[2]; units_file <- args[3]
start <- as.integer(Sys.getenv("UNIT_START", "0"))
end <- as.integer(Sys.getenv("UNIT_END", "-1"))
LEVELS <- c(0.05, 0.10, 0.15, 0.25, 0.40)

units <- read.csv(units_file, stringsAsFactors = FALSE)   # name, K, rseed
if (end < 0) end <- nrow(units)

bisq_one <- function(formula, data, nc) {
  f <- get("mixlinrb_bione", envir = asNamespace("RobMixReg"))
  body_txt <- deparse(body(f))
  body_txt <- sub("tmp_mod = lm(formula, data = data, weights = w)",
                  "{environment(formula) <- environment(); tmp_mod = lm(formula, data = data, weights = w)}",
                  body_txt, fixed = TRUE)
  g <- f; body(g) <- parse(text = body_txt)[[1]]; environment(g) <- asNamespace("RobMixReg")
  g(formula, data, nc)
}
bisq_patched <- function(formula, data, nc = 2, nit = 20) {
  f <- getMethod("mixlinrb_bi", signature("formula", "ANY", "numeric", "numeric"))@.Data
  body_txt <- deparse(body(f))
  body_txt <- gsub("mixlinrb_bione(formula, data, nc)", "bisq_one(formula, data, nc)",
                   body_txt, fixed = TRUE)
  g <- f; body(g) <- parse(text = body_txt)[[1]]
  e <- new.env(parent = asNamespace("RobMixReg")); e$bisq_one <- bisq_one
  environment(g) <- e
  g(formula, data, nc, nit)
}

coefs_from <- function(cc, d, K) {
  # cc: rows = coefficients (first d rows), columns = components
  cc <- as.matrix(cc)
  if (ncol(cc) != K || nrow(cc) < d) return(rep(NA_real_, K * d))
  as.vector(cc[1:d, , drop = FALSE])      # column-major: b_1_1..b_1_d, b_2_1, ...
}

rows <- list()
for (i in (start + 1):end) {
  u <- units[i, ]
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
  for (a in LEVELS) {
    run(sprintf("tcwm(%.2f)", a), function() trim.cwm(X = X, Y = dat$y, K = K, alpha = a, niter = 50, Ksteps = 20),
        function(r) mean(r@ctleclusters == 0))
    run(sprintf("TLE(%.2f)", a), function() TLE(y ~ ., dd, nc = K, tRatio = a, MaxIt = 200),
        function(r) mean(r@ctleclusters == -1))
  }
  run("CTLE", function() CTLERob(y ~ ., dd, nc = K), function(r) length(r@indout) / nrow(dd))
  run("bisq", function() bisq_patched(y ~ ., dd, nc = K, nit = 20), function(r) 0)
}
df <- do.call(rbind, lapply(rows, function(r) as.data.frame(r, stringsAsFactors = FALSE)))
write.csv(df, out_file, row.names = FALSE)
cat(sprintf("UNITS_DONE=%d\n", end - start))
