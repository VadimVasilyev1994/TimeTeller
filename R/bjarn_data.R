#' Bjarnasson microarray data
#'
#' A dataset containing oral mucosa expression profiles from 10 healthy individuals taken at 6 different time points 4 hours apart.
#' For memory and speed considerations 1000 randomly selected probes together with clock probes were selected (to be removed).
#'
#' @format A list containing the following:
#' \describe{
#'  \item{expr_mat}{fRMA normalised expression matrix}
#'  \item{group}{Individual (order corresponds to columns in the expression matrix)}
#'  \item{time}{Actual sampling time (order corresponds to columns in the expression matrix)}
#'  \item{probes_used}{Clock gene probes used for training the model}
#' }
#' @source Bjarnasson Oral Mucosa
"bjarn_data"
