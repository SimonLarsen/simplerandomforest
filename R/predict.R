#' Predict on new data with a simplerandomforest model.
#' 
#' @param model A \code{simplerandomforest} object.
#' @param x Data for prediction.
#' @param num.threads Number of threads to use. Default is number of cores available.
#' @param ... Further arguments passed to and from other methods.
#' @export
predict <- function(model, x, num.threads=NULL, ...) UseMethod("predict")

#' @export
predict.simplerandomforest <- function(model, x, num.threads=NULL, ...) {
  if(is.null(num.threads)) num.threads <- 0
  
  x <- data.matrix(x)
  
  out <- predictCpp(model, x, length(model$levels), num.threads)
  pred <- factor(model$levels[out+1], levels=model$levels)
  
  return(pred)
}
