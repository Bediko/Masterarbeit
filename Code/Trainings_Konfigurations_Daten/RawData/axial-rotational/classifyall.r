#
# Evaluate classification axial versus rotational symmetry
# Version with all features (egde_dir, skel_size, anti_par, cov_ratio)
#

#source("gaussian.bayes.r")

#==========================================================
# Arguments:
#   data = data frame (one feature vector per line)
#   dataclasses = classes of the lines in the data frame
#
# Returns:
#   creates a colored scatter plot
#
scatter.plot <- function(data, dataclasses, title=NA) {
  if (ncol(data) > 2) {
    stop("can only plot 2d data")
  }

  classes <- levels(factor(dataclasses))
  minx = min(data[,1])-0.5
  maxx = max(data[,1])+1.0
  miny = min(data[,2])-0.5
  maxy = max(data[,2])+1.0
  for (i in 1:length(classes)) {
    cl = classes[i]
    if (i == 1) {
      plot(data[dataclasses==cl,1], data[dataclasses==cl,2], xlim=c(minx,maxx), ylim=c(miny,maxy), col=i, main=title)
    } else {
      points(data[dataclasses==cl,1], data[dataclasses==cl,2], col=i)
    }
  }
  legend("topright",classes,lty=c(1,1,1,1),col=1:length(classes))
}


library(MASS)
library(kernlab)
library(lattice)
library(tree)

file <- "E:\\Hochschule Niederrhein\\Semester6\\NN\\data\\axial-rotational\\feature_aor_all_features_p_0.5.csv"

load_data <- function()
{
    return(read.csv(file, header=FALSE, col.names=c("class","edge_dir","skel_size","anti_par","cov_ratio")))
}
x <- load_data()
#x <- x[,-5]

# scatter plot matrix
spl <- splom(~x[,-1], groups=x$class, data=x)
print(spl)


# baseline Recognition Rate: old rule based on edge directedness alone
Na <- sum(x$class == 'a')
Nr <- sum(x$class == 'r')
cat("Baseline (edge_dir > 0.27):\n")
cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
    sum(x$class=='a' & x$edge_dir < 0.27) / Na,
    sum(x$class=='r' & x$edge_dir < 0.27) / Nr))

# simple two rule decision
cat("\nSimple two rule decision:\n")
skel_t <- max(x$skel_size[x$class=='r'])
skel_t = 0.46
z <- x$class
z[x$edge_dir < 0.27] <- 'r'
z[x$edge_dir >= 0.27] <- 'a'
z[x$skel_size > skel_t] <- 'a' 
cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
    sum(x[1]=='a' & z == 'r') / Na,  sum(x[1]=='r' & z == 'r') / Nr))

# recognition rate of LDA
cat("\nLDA with all:\n")
z <- lda(class ~ edge_dir + skel_size + anti_par + cov_ratio, x, prior=c(0.5,0.5))
#z <- lda(class ~ edge_dir + skel_size + anti_par, x, prior=c(0.5,0.5))
lda.class <- predict(z, x, prior=c(0.5,0.5))$class
cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
    sum(x[1]=='a' & lda.class == 'r') / Na,  sum(x[1]=='r' & lda.class == 'r') / Nr))

# LDA after threshold on skel_size
cat("\n1) skel_size 2) LDA on rest:\n")
skel_t <- max(x$skel_size[x$class=='r'])
y <- x[x$skel_size <= skel_t,]
n_filtered_out <- nrow(x) - nrow(y)
z <- lda(class ~ edge_dir + skel_size + anti_par + cov_ratio, y, prior=c(0.5,0.5))
#z <- lda(class ~ edge_dir + skel_size + anti_par, y, prior=c(0.5,0.5))
lda.class <- predict(z, y, prior=c(0.5,0.5))$class
cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
    sum(y[1]=='a' & lda.class == 'r') / Na,  sum(y[1]=='r' & lda.class == 'r') / Nr))

# recognition rate of QDA
cat("\nQDA with all:\n")
z <- qda(class ~ edge_dir + skel_size + anti_par + cov_ratio, x, method='t', prior=c(0.5,0.5))
#z <- qda(class ~ edge_dir + skel_size + anti_par, x, method='t', prior=c(0.5,0.5))
qda.class <- predict(z, x, prior=c(0.5,0.5))$class
cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
    sum(x$class=='a' & qda.class == 'r') / Na,  sum(x$class=='r' & qda.class == 'r') / Nr))

# QDA after threshold on skel_size
cat("\n1) skel_size 2) QDA on rest:\n")
skel_t <- max(x$skel_size[x$class=='r'])
y <- x[x$skel_size <= skel_t,]
n_filtered_out <- nrow(x) - nrow(y)
#z <- qda(class ~ edge_dir + anti_par + cov_ratio, y, method='t',prior=c(0.5,0.5))
z <- qda(class ~ edge_dir + anti_par, y, method='t',prior=c(0.5,0.5))
qda.class <- predict(z, y, prior=c(0.5,0.5))$class
cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
    sum(y[1]=='a' & qda.class == 'r') / Na,  sum(y[1]=='r' & qda.class == 'r') / Nr))

# Gaussian bayes with all features
#cat("\nGaussian Bayes:\n")
#gb.class = gaussian.bayes(x[,2:5], x[,2:5], x$class, aprior='equal')
#gb.class = gaussian.bayes(x[,2:4], x[,2:4], x$class)
#cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
#    sum(x$class=='a' & gb.class == 'r') / Na,  sum(x$class=='r' & gb.class == 'r') / Nr))

# Decision tree
cat("\nDecision tree:\n")
t <- tree(class ~ edge_dir + skel_size + anti_par + cov_ratio, x)
#t <- tree(class ~ edge_dir + skel_size + anti_par, x)
t.pruned <- prune.tree(t, best=4)
t.class <- predict(t.pruned, x, type="class")
cat(sprintf("  a_as_r=%5.4f  r_as_r=%5.4f\n",
    sum(x$class=='a' & t.class == 'r') / Na,  sum(x$class=='r' & t.class == 'r') / Nr))
print(t.pruned)


