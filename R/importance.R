#' Get variable importance.
#' 
#' @param x A \code{simplerandomforest} object.
#' @param ... Further arguments passed down to and from other methods.
#' @export
importance <- function(x, ...) UseMethod("importance")

#' @export
importance.simplerandomforest <- function(x, ...) {
  x$importance
}
