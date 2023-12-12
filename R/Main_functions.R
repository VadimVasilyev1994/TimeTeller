#' Train the TimeTeller model
#'
#' Trains the model using provided expression matrix and metadata.
#'
#' @param exp_matrix Matrix or data frame with features in rows and samples in columns
#' @param genes genes used to construct the timeteller model
#' @param group_1 vector containing metadata for each sample (eg individual, organ)
#' @param group_2 vector containing metadata for each sample (eg individual, organ). Leave empty if unavailable
#' @param group_3 vector containing metadata for each sample (eg individual, organ). Leave empty if unavailable
#' @param time vector containing timing of each sample. Please appreciate the difference between CT and ZT
#' @param replicate vector containing replicate information if available
#' @param mat_normalised was the expression matrix supplied already normalised (eg RMA for microarray, CPM/TPM for RNA, etc) or not. If not, \code{edgeR::cpm} function will be used (pseudocount of 2) to obtain log normalised values for downstream analysis. Default is TRUE
#' @param treat_independently should replicates be averaged (might be useful for technical replicates, not recommended for biological replicates). Default is TRUE
#' @param combine_for_norm should replicates be combined for normalisation. If TRUE, replicates will be combined for normalisation purposes (eg in \code{timecourse} normalisation, replicates will be combined and treated as one timeseries). Default is FALSE
#' @param parallel_comp if TRUE, parallel computation using \code{foreach} and \code{doParallel} packages will be used. Default is FALSE
#' @param cores if using parallel computation, how many cores should be used. Default is 4
#' @param method method used for normalisation. Must be one of the \code{'intergene'}, \code{'timecourse'}, \code{'timecourse_matched'}, \code{'clr'}
#' @param grouping_vars group variables (eg \code{'group_1'}, \code{'group_2'} or their combination \code{c('group_1','group_2')}) used for \code{timecourse_matched} normalisation
#' @param num_PC how many principal components should be used for the analysis. Default is 3
#' @param log_thresh log threshold selected. This is an important parameter and should be chosen carefully. Please read \code{help(choose_logthresh_plot)} for further information
#' @param epsilon hyperparameter value used in theta calculation. No reason to change this, more information at \url{https://www.biorxiv.org/content/10.1101/2023.03.14.532177v1}. Default is 0.40
#' @param eta hyperparameter value used in theta calculation. No reason to change this, more information at \url{https://www.biorxiv.org/content/10.1101/2023.03.14.532177v1}. Default is 0.35
#' @param cov_estimate type of covariance estimate used. If \code{'normal'} is selected, \code{Rfast::mvnorm.mle} will be used. If \code{'robust'} is selected, \code{rrcov::CovMcd} will be used instead. This can be useful when there is a lot of training data, some of which is suspected to have high technical noise. Since we don't want to ignore biological noise (eg chronotype) and it's often hard to tell the two apart, default is \code{'normal'}
#' @param alpha_par this is following \code{rrcov} package (in case \code{'robust'} was chosen above) and is the parameter controlling the size of the subsets over which the determinant will be minimized. Allowed values are between 0.5 and 1
#' @param num_interp_points number of interpolation points to use for the mean spline and covariance matrices. Default is 144 and this corresponds to 24 * 60 / 144 = 10 minutes
#' @param interp_method method used for spline interpolation. Must be one of the \code{'perpchip'} (which uses an adapted periodic version of \code{pchip} from \code{pracma} package) or \code{'standard'} (which uses periodic version of standard \code{spline} function from \code{splines} package).
#' @param cov_path method for covariance matrix interpolation. Must be one of the \code{'spline'} (in which case covariance matrix will be interpolated element-wise using \code{interp_method} selection) or \code{'fisherrao'} (in which case the geodesics based on the Fisher–Rao metric for Gaussian distributions will be used).
#' @param minpeakheight this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development. Please consult \code{?pracma::findpeaks}, if needed, for further information
#' @param minpeakdistance this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param nups this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param ndowns this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param threshold this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param npeaks this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#'
#'
#' @author Vadim Vasilyev
#'
#' @import dplyr
#' @import purrr
#' @import tibble
#' @importFrom tidyr unite pivot_longer
#' @import ggplot2
#' @import foreach
#' @import doParallel
#' @importFrom pracma findpeaks
#' @import splines
#' @importFrom compositions clr
#' @import rgl
#' @import plotly
#' @import cosinor2
#' @importFrom Rfast mvnorm.mle
#' @import utils
#' @import stats
#' @import grDevices
#' @import graphics
#' @import Matrix
#' @import edgeR
#' @import SummarizedExperiment
#' @import circular
#' @import gprofiler2
#' @importFrom data.table transpose
#' @import ggrepel
#' @import mvtnorm
#' @import psych
#' @import rrcov
#'
#'
#' @return Returned is the rich object of class \code{list} containing the TimeTeller model for further analysis
#' @export
#'
#' @examples
#' load(bjarn_data)
#' tt_model <- train_model(exp_matrix = bjarn_data$expr_mat, genes = bjarn_data$probes_used, group_1 = bjarn_data$group, time = bjarn_data$time, log_thresh = -5)
#' tt_model <- train_model(exp_matrix = bjarn_data$expr_mat, genes = bjarn_data$probes_used, group_1 = bjarn_data$group, time = bjarn_data$time, method = 'timecourse', parallel_comp = TRUE, log_thresh = -5)
train_model <- function(exp_matrix, genes, group_1, group_2, group_3, time, replicate, mat_normalised = TRUE,
                        treat_independently = TRUE, combine_for_norm = FALSE, parallel_comp = FALSE, cores = 4,
                        method = 'intergene', grouping_vars = c('Group'), num_PC = 3, log_thresh, epsilon = 0.4, eta = 0.35,
                        cov_estimate = 'normal', alpha_par = 0.75, num_interp_points = 144, interp_method = 'perpchip', cov_path = 'spline',
                        minpeakheight = -Inf, minpeakdistance = 1, nups = 1, ndowns = 0, threshold = 0, npeaks = 2) {
  if(parallel_comp) {
    my.cluster <- parallel::makeCluster(
      cores,
      type = "PSOCK"
    )
    doParallel::registerDoParallel(cl = my.cluster)
    message('Using parallel computation with N=',cores)
  }
  object <- make_data_object(exp_matrix = exp_matrix, genes = genes, group_1 = group_1, group_2 = group_2, group_3 = group_3, time = time, replicate = replicate, mat_normalised = mat_normalised)
  object <- deal_with_replicates(object, treat_independently = treat_independently)
  object <- allocate_timeseries_ind(object, combine_for_norm = combine_for_norm)
  object <- normalised_df(object, method = method, grouping_vars = grouping_vars)
  message('Data loaded and successfully normalised')
  message('Calculating projections and interpolating densities...')
  object <- get_projections(object, num_PC = num_PC)
  object <- get_mvn_original_data(object, cov_estimate = cov_estimate, alpha_par = alpha_par)
  object <- get_mvn_interpolated(object, num_interp_points = num_interp_points, interp_method = interp_method, cov_path = cov_path)
  message('Calculating likelihoods...')
  if(parallel_comp) {
    object <- calc_train_likelis_dev(object)
  } else {
    object <- calc_train_likelis(object)
  }
  object <- get_final_likelis_train(object, log_thresh = log_thresh)
  if(parallel_comp) {
    object <- theta_calc_train_dev(object, epsilon = epsilon, eta = eta)
  } else {
    object <- theta_calc_train(object, epsilon = epsilon, eta = eta)
  }
  message('Calculating additional info...')
  if(parallel_comp) {
    object <- calc_flat_theta_contrib_train_dev(object)
  } else {
    object <- calc_flat_theta_contrib_train(object)
  }
  object <- second_peaks_fun_train(object, minpeakheight = minpeakheight, minpeakdistance = minpeakdistance, nups = nups, ndowns = ndowns, threshold = threshold, npeaks = npeaks)
  message('Finished!')
  if(parallel_comp) {
    parallel::stopCluster(cl = my.cluster)
  }
  return(object)
}

train_cv <- function(group_to_leave_out = 'group_1', genes, exp_matrix, test_grouping_vars, group_1, group_2, group_3, time, replicate, mat_normalised = TRUE,
                     treat_independently = TRUE, combine_for_norm = FALSE, parallel_comp = FALSE, cores = 4,
                     method = 'intergene', grouping_vars = c('Group'), num_PC = 3, log_thresh, epsilon = 0.4, eta = 0.35,
                     cov_estimate = 'normal', alpha_par = 0.75, num_interp_points = 144, interp_method = 'perpchip', cov_path = 'spline',
                     minpeakheight = -Inf, minpeakdistance = 1, nups = 1, ndowns = 0, threshold = 0, npeaks = 2) {
  if (missing(group_2)) {group_2 <- as.character(rep(NA, length(group_1)))}
  if (missing(group_3)) {group_3 <- as.character(rep(NA, length(group_1)))}
  if (missing(replicate)) {replicate <- as.character(rep(NA, length(group_1)))}
  if (missing(time)) {stop("Model can't be trained without training times. Please provide time vector")}
  group_cv <- get(group_to_leave_out)
  cv_res <- list()
  groupcv <- unique(group_cv)
  for (i in 1:length(groupcv)) {
    message('Leaving out ',groupcv[i])
    ind_train <- which(!group_cv %in% groupcv[i])
    if(any(table(time[ind_train]) < 3)) {
      message('Some times have 2 or less observations. Those will be removed for training')
      ind_enough_times <- which(!time[ind_train] %in% as.numeric(names(table(time[ind_train])[table(time[ind_train]) < 3])))
    } else {
      ind_enough_times <- 1:dim(exp_matrix[,ind_train])[2]
    }
    trained_model <- suppressMessages(quiet(train_model(exp_matrix = exp_matrix[,ind_train][,ind_enough_times], genes = genes,
                                                        group_1 = group_1[ind_train][ind_enough_times], group_2 = group_2[ind_train][ind_enough_times], group_3 = group_3[ind_train][ind_enough_times],
                                                        time = time[ind_train][ind_enough_times], replicate = replicate[ind_train][ind_enough_times], mat_normalised = mat_normalised,
                                                        treat_independently = treat_independently, combine_for_norm = combine_for_norm, parallel_comp = parallel_comp, cores = cores,
                                                        method = method, grouping_vars = grouping_vars, num_PC = num_PC, log_thresh = log_thresh, epsilon = epsilon, eta = eta,
                                                        cov_estimate = cov_estimate, alpha_par = alpha_par, num_interp_points = num_interp_points, interp_method = interp_method, cov_path = cov_path,
                                                        minpeakheight = minpeakheight, minpeakdistance = minpeakdistance, nups = nups, ndowns = ndowns, threshold = threshold, npeaks = npeaks)))

    ind_test <- which(group_cv %in% groupcv[i])
    cv_res[[i]] <- suppressMessages(quiet(test_model(trained_model, exp_matrix = exp_matrix[,ind_test], test_grouping_vars = test_grouping_vars,
                                                     test_group_1 = group_1[ind_test], test_group_2 = group_2[ind_test], test_group_3 = group_3[ind_test],
                                                     test_time = time[ind_test], test_replicate = replicate[ind_test], mat_normalised_test = mat_normalised, log_thresh = log_thresh,
                                                     parallel_comp = parallel_comp, cores = cores,
                                                     minpeakheight = minpeakheight, minpeakdistance = minpeakdistance, nups = nups, ndowns = ndowns, threshold = threshold, npeaks = npeaks)))
    message('Finished ',groupcv[i])
  }
  names(cv_res) <- groupcv
  return(cv_res)
}


#' Project test data on the training model
#'
#' Projects the test data onto the model obtained after running \code{train_model} function.
#' Among the outputs are the estimated time for each sample and the clock dysfunction metric (Theta)
#'
#' @param object object of class list containing timeteller training model (obtained after running \code{train_model} function)
#' @param exp_matrix matrix or data frame containing test data with features in rows and samples in columns
#' @param test_grouping_vars groups below that will be used for \code{timecourse_matched} normalisation. See the \code{vignette} for examples
#' @param test_group_1 vector containing metadata for each sample (eg individual, organ). These are not required for \code{intergene} normalisation, however may be useful for visualisation. For \code{timecourse} and \code{timecourse_matched} normalisations, however, these are important and should be supplied in the same format as for the training data to ensure consistent results
#' @param test_group_2 vector containing metadata for each sample (eg individual, organ)
#' @param test_group_3 vector containing metadata for each sample (eg individual, organ)
#' @param test_replicate vector containing replicate information if available
#' @param test_time vector containing replicate information if available. This does not affect the results but is useful to calculate the prediction error and hence the algorithm accuracy
#' @param mat_normalised_test was the expression matrix supplied already normalised (eg RMA for microarray, CPM/TPM for RNA, etc) or not. If not, \code{edgeR::cpm} function will be used (pseudocount of 2) to obtain log normalised values for downstream analysis. Default is TRUE
#' @param log_thresh log threshold selected. This is an important parameter and should be chosen carefully.Please read \code{help(choose_logthresh_plot)} for further information
#' @param parallel_comp if TRUE, parallel computation using \code{foreach} and \code{doParallel} packages will be used. Default is FALSE
#' @param cores if using parallel computation, how many cores should be used. Default is 4
#' @param minpeakheight this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development. Please consult \code{?pracma::findpeaks}, if needed, for further information
#' @param minpeakdistance this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param nups this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param ndowns this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param threshold this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#' @param npeaks this is used internally with \code{pracma} package to locate peaks. No reason to change, however included for future testing / development
#'
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returned is the rich object of class \code{list} containing the results for both train and test models.
#' @export
#'

test_model <- function(object, exp_matrix, test_grouping_vars, test_group_1, test_group_2, test_group_3, test_replicate, test_time, mat_normalised_test = TRUE,
                       log_thresh, parallel_comp = FALSE, cores = 4, minpeakheight = -Inf, minpeakdistance = 1, nups = 1, ndowns = 0, threshold = 0, npeaks = 2) {
  if(parallel_comp) {
    my.cluster <- parallel::makeCluster(
      cores,
      type = "PSOCK"
    )
    doParallel::registerDoParallel(cl = my.cluster)
    message('Using parallel computation with N=',cores)
  }
  object <- add_test_data(object, exp_matrix = exp_matrix, test_grouping_vars = test_grouping_vars, test_group_1 = test_group_1, test_group_2 = test_group_2, test_group_3 = test_group_3, test_replicate = test_replicate, test_time = test_time, mat_normalised_test = mat_normalised_test)
  message('Test data loaded and successfully normalised')
  message('Calculating likelihoods...')
  if(parallel_comp) {
    object <- calc_test_likelis_dev(object)
  } else {
    object <- calc_test_likelis(object)
  }
  object <- get_final_likelis_test(object, log_thresh = log_thresh)
  if(parallel_comp) {
    object <- theta_calc_test_dev(object)
  } else {
    object <- theta_calc_test(object)
  }
  message('Calculating additional info...')
  if(parallel_comp) {
    object <- calc_flat_theta_contrib_test_dev(object)
  } else {
    object <- calc_flat_theta_contrib_test(object)
  }
  object <- second_peaks_fun_test(object, minpeakheight = minpeakheight, minpeakdistance = minpeakdistance, nups = nups, ndowns = ndowns, threshold = threshold, npeaks = npeaks)
  message('Finished!')
  if(parallel_comp) {
    parallel::stopCluster(cl = my.cluster)
  }
  return(object)
}
