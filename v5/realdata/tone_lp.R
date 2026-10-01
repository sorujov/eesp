suppressMessages(library(RobMixReg))
out <- list()
for (f in c("tone_clean","tone_out")) {
 d <- read.csv(paste0(f,".csv")); names(d) <- c("x1","y")
 for (s in 1:10) { set.seed(s)
  r <- tryCatch(mixLp(y ~ x1, d, nc = 2, nit = 20), error = function(e) NULL)
  if (!is.null(r)) out[[length(out)+1]] <- data.frame(data=f, method="Laplace", seed=s, t(as.vector(r@compcoef[1:2,])), trimmed=0)
 }
}
write.csv(do.call(rbind, out), "tone_lp.csv", row.names = FALSE)
