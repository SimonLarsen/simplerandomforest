#' Graph-guided Random Forest.
#' 
#' @param x Training data.
#' @param y Reponse variables.
#' @param num.trees Number of decision trees to train.
#' @param mtry Number of variable to consider at each split. Default is square root of number of independent variables.
#' @param replace Sample with replacement.
#' @param sample.fraction Fraction of observations to sample.
#' @param permutation.importance Compute permutation importance.
#' @param num.threads Number of threads to use. Default is number of cores available.
#' @export
simplerandomforest <- function(
  x, y,
  num.trees = 500,
  mtry = NULL,
  replace = FALSE,
  sample.fraction = ifelse(replace, 1, 0.632),
  permutation.importance = FALSE,
  num.threads = NULL
) {
  if(class(y) != "factor") {
    stop("Response variables must be of type `factor`.")
  }
  if(length(y) != nrow(x)) {
    stop("Number of reponse variables does not match number of observations.")
  }
  if(is.null(mtry)) {
    mtry = ceiling(sqrt(ncol(x)))
  }
  if(is.null(num.threads)) {
    num.threads = 0
  }
  
  x <- data.matrix(x)
  y.numeric <- as.numeric(y)-1
  
  result <- simplerandomforestCpp(x, y.numeric, num.trees, mtry, replace, sample.fraction, permutation.importance, num.threads)
  
  result$num.observations <- nrow(x)
  result$num.features <- ncol(x)
  result$distribution <- table(y)
  result$levels <- levels(y)
  result$num.trees <- num.trees
  result$mtry <- mtry
  result$replace <- replace
  result$sample.fraction <- sample.fraction
  result$gini.importance <- stats::setNames(as.vector(result$gini.importance), colnames(x))
  if(permutation.importance) result$permutation.importance <- stats::setNames(as.vector(result$permutation.importance), colnames(x))
  
  class(result) <- "simplerandomforest"
  return(result)
}
