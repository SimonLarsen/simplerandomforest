#' Print description of simplerandomforest object.
#' 
#' @param x A \code{simplerandomforest} object.
#' @param ... Further arguments passed down or to from other methods.
#' @export
print.simplerandomforest <- function(x, ...) {
  cat("Random Forest model.\n")
  cat("Number of observations: ", x$num.observations, "\n")
  cat("Number of features:     ", x$num.features, "\n")
  cat("Number of timepoints:   ", x$num.timepoints, "\n")
  cat("Number of trees:        ", x$num.trees, "\n")
  cat("mtry:                   ", x$mtry, "\n")
  cat("replace:                ", x$replace, "\n")
  cat("sample.fraction:        ", x$sample.fraction, "\n")
  cat("OOB error rate:         ", x$oob.error, "\n")
}
