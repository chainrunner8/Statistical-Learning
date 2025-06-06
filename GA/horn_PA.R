setwd(r"(\informatics\python\Leiden\Statistical-Learning\GA)")

X_scaled = read.csv('X_scaled.csv', row.names=1)
head(X_scaled)

ev <- eigen(cor(X_scaled))
ap <- parallel(subject=nrow(X_scaled), var=ncol(X_scaled), rep=1000)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS,main="")