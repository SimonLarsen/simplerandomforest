// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// predictCpp
arma::uvec predictCpp(const Rcpp::List& model, const arma::mat& x, unsigned int y_levels, size_t num_threads);
RcppExport SEXP _simplerandomforest_predictCpp(SEXP modelSEXP, SEXP xSEXP, SEXP y_levelsSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type model(modelSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type y_levels(y_levelsSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(predictCpp(model, x, y_levels, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// simplerandomforestCpp
List simplerandomforestCpp(const arma::mat& x, const arma::uvec& y, size_t num_trees, size_t mtry, bool replace, double sample_fraction, bool permutation_importance, size_t num_threads);
RcppExport SEXP _simplerandomforest_simplerandomforestCpp(SEXP xSEXP, SEXP ySEXP, SEXP num_treesSEXP, SEXP mtrySEXP, SEXP replaceSEXP, SEXP sample_fractionSEXP, SEXP permutation_importanceSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< size_t >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< size_t >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< double >::type sample_fraction(sample_fractionSEXP);
    Rcpp::traits::input_parameter< bool >::type permutation_importance(permutation_importanceSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(simplerandomforestCpp(x, y, num_trees, mtry, replace, sample_fraction, permutation_importance, num_threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_simplerandomforest_predictCpp", (DL_FUNC) &_simplerandomforest_predictCpp, 4},
    {"_simplerandomforest_simplerandomforestCpp", (DL_FUNC) &_simplerandomforest_simplerandomforestCpp, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_simplerandomforest(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
