suppressMessages(library(RobMixReg))
d <- read.csv("fishery_log.csv"); X <- as.matrix(d$x1)
out <- list()
for (s in 1:10) {
  set.seed(s)
  r <- tryCatch(CTLERob(y ~ x1, d, nc = 2), error = function(e) NULL)
  if (!is.null(r)) out[[length(out)+1]] <- data.frame(method="CTLE", seed=s, t(as.vector(r@compcoef[1:2,])), trimmed=length(r@indout)/nrow(d))
  for (a in c(0.04, 0.10, 0.25)) {
    r <- tryCatch(trim.cwm(X = X, Y = d$y, K = 2, alpha = a, niter = 50, Ksteps = 20), error = function(e) NULL)
    if (!is.null(r)) out[[length(out)+1]] <- data.frame(method=sprintf("tcwm(%.2f)",a), seed=s, t(as.vector(r@compcoef[1:2,])), trimmed=mean(r@ctleclusters==0))
  }
}
write.csv(do.call(rbind, out), "fish_r.csv", row.names = FALSE)
