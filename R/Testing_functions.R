normalise_test_data <- function(object, exp_matrix, test_grouping_vars, test_group_1, test_group_2, test_group_3, test_replicate, test_time) {

  norm_method <- object[['Normalisation_choice']]

  if (norm_method == 'intergene') {
    normalised_mat <- apply(exp_matrix, 2, function(x) {(x - mean(x)) / sd(x)})
  }

  if (norm_method == 'clr') {
    normalised_mat <- apply(exp_matrix,2,compositions::clr)
  }

  else if(norm_method == 'timecourse') {
    splitting_vec <- paste0(tidyr::replace_na(test_group_1,''), tidyr::replace_na(test_group_2,''), tidyr::replace_na(test_group_3,''), tidyr::replace_na(test_replicate,''))
    if(all(sapply(splitting_vec, function(x) x==''))) {stop("No groups specified for timecourse normalisation.")}
    split_df <- split(as.data.frame(t(exp_matrix)), splitting_vec)
    split_df_ts_norm <- lapply(split_df, function(x) as.data.frame(scale(x)))
    df_normalised <- unsplit(split_df_ts_norm, splitting_vec)
    normalised_mat <- t(df_normalised)
  }

  else if(norm_method == 'timecourse_matched') {
    group_metrics_list <- object[['Train']][['Gene_Per_Group_Info']]
    means <- group_metrics_list$Mean
    sds <- group_metrics_list$SD
    splitting_vec <- data.frame(Test_Group_1 = test_group_1, Test_Group_2 = test_group_2, Test_Group_3 = test_group_3, Test_Replicate = test_replicate) %>%
      tidyr::unite('test_timeseries_matched_name', all_of(test_grouping_vars), remove = FALSE, sep = '|') %>% dplyr::pull(test_timeseries_matched_name)
    split_df <- split(as.data.frame(t(exp_matrix[object$Metadata$Train$Genes_Used, ])), splitting_vec)
    object[['Check']] <- split_df
    df_names <- names(split_df)
    cat(paste('# Data found for the following groups: ', names(split_df)[names(split_df) %in% rownames(means)],'\n'))
    cat(paste('# Data NOT found for the following groups: ', names(split_df)[!(names(split_df) %in% rownames(means))],' ==> For these standard timecourse normalisaiton will be used\n'))
    split_df_ts_norm <- lapply(seq_along(split_df), function(i)
      as.data.frame(scale(split_df[[i]], center = (if (names(split_df)[i] %in% rownames(means)) as.vector(t(means[names(split_df)[i],])) else TRUE),
                          scale = TRUE)))
    df_normalised <- unsplit(split_df_ts_norm, splitting_vec)
    normalised_mat <- t(df_normalised)
  }

  return(normalised_mat)
}

add_test_data <- function(object, exp_matrix, test_grouping_vars, test_group_1, test_group_2, test_group_3, test_replicate, test_time, mat_normalised_test) {
  if(missing(test_group_1))   {test_group_1 <- as.character(rep(NA, dim(exp_matrix)[2]))}
  if(missing(test_group_2))   {test_group_2 <- as.character(rep(NA, dim(exp_matrix)[2]))}
  if(missing(test_group_3))   {test_group_3 <- as.character(rep(NA, dim(exp_matrix)[2]))}
  if(missing(test_replicate)) {test_replicate <- as.character(rep(NA, dim(exp_matrix)[2]))}
  if(missing(test_time))      {test_time <- as.character(rep(NA, dim(exp_matrix)[2]))}

  ll <- list(test_group_1,test_group_2,test_group_3,test_time,test_replicate)
  if (!all(sapply(ll,length)==length(ll[[1]]))) {stop("Supplied metadata is not all of equal length. Please check inputs")}
  if (!all(object[['Metadata']][['Train']][['Genes_Used']] %in% rownames(exp_matrix))) {stop("Not all training genes found in the supplied data. Please check inputs")}

  genes_used <- object[['Metadata']][['Train']][['Genes_Used']]
  if (mat_normalised_test) {
    object[['Test_Data']][['Full_Test_Data_Raw']] <- 'Not Provided'
    object[['Test_Data']][['Full_Test_Data']] <- as.matrix(exp_matrix)
  } else {
    object[['Test_Data']][['Full_Test_Data_Raw']] <- as.matrix(exp_matrix)
    object[['Test_Data']][['Full_Test_Data']] <- prepare_raw_counts(exp_matrix)
  }
  test_data <- as.matrix(exp_matrix[genes_used,])
  object[['Test_Data']][['Test_Exp_Data']] <- test_data
  object[['Test_Data']][['Normalised_Test_Exp_Data']] <- normalise_test_data(object = object, exp_matrix = test_data, test_grouping_vars = test_grouping_vars, test_group_1 = test_group_1, test_group_2 = test_group_2, test_group_3 = test_group_3, test_replicate = test_replicate, test_time = test_time)

  object[['Metadata']][['Test']][['Group_1']] <- test_group_1
  object[['Metadata']][['Test']][['Group_2']] <- test_group_2
  object[['Metadata']][['Test']][['Group_3']] <- test_group_3
  object[['Metadata']][['Test']][['Time']] <- test_time
  object[['Metadata']][['Test']][['Replicate']] <- test_replicate

  return(object)
}

calc_test_likelis <- function(object) {
  svd_data <- object[['Projections']][['SVD_Per_Time_Point']]
  fitted_mvn_data <- object[['Projections']][['Fitted_MVN_Interpolated']]
  test_exp_data <- object[['Test_Data']][['Normalised_Test_Exp_Data']]
  num_PC <- object[['PC_Num']]
  test_likelihood_array <- base::array(data = NA, dim = c(dim(fitted_mvn_data)[2], dim(test_exp_data)[2], dim(fitted_mvn_data)[3]))
  test_projections <- list()
  for (i in 1:length(names(svd_data))) {
    project_exp_mat <- svd_data[[i]] %*% test_exp_data
    test_projections[[i]] <- project_exp_mat
    for (ind_num in 1:dim(test_exp_data)[2]) {
      vec <- c()
      for(j in 1:dim(fitted_mvn_data)[2]) {
        curr_sigma <- matrix(fitted_mvn_data[(num_PC+1):(num_PC+num_PC^2),j,i], nrow = num_PC)
        curr_eig <- eigen(curr_sigma)$values
        if (any(curr_eig < 0)) {
          sigma_used <- nearPD(curr_sigma, base.matrix = TRUE, ensureSymmetry = TRUE, eig.tol = 1e-05, conv.tol = 1e-06, posd.tol = 1e-07)$mat
        } else {
          sigma_used <- curr_sigma
        }

        vec[j] <- mvtnorm::dmvnorm(project_exp_mat[,ind_num], mean = fitted_mvn_data[1:num_PC,j,i], sigma = sigma_used, checkSymmetry = FALSE)

      }
      test_likelihood_array[,ind_num,i] <- vec
    }
    Sys.sleep(0.01)
    cat("\rFinished", i, "of", length(names(svd_data)), "\n")
  }
  names(test_projections) <- names(svd_data)
  object[['Test_Data']][['Test_Projections']] <- test_projections
  object[['Test_Data']][['Test_Likelihood_Array']] <- log(test_likelihood_array)
  return(object)
}

get_final_likelis_test <- function(object, log_thresh){
  likelis_array <- object[['Test_Data']][['Test_Likelihood_Array']]
  object[['Test_Data']][['LogThresh_Test']] <- log_thresh
  num_time_points <- dim(likelis_array)[3]
  for (i in 1:num_time_points) {
    likelis_array[,,i] <- pmax(likelis_array[,,i], log_thresh)
  }
  object[['Test_Data']][['Likelis_Post_Thresh_Test']] <- likelis_array
  final_averaged_likelis <- apply(likelis_array, c(1,2), mean)
  object[['Test_Data']][['Averaged_Likelis_Post_Thresh_Test']] <- final_averaged_likelis
  max_likelis <- apply(final_averaged_likelis,2,max)
  object[['Test_Data']][['Max_Likelis_Test']] <- max_likelis
  return(object)
}

theta_calc_test <- function(object) {
  epsilon <- object[['Train_Data']][['epsilon']]
  eta <- object[['Train_Data']][['eta']]
  averaged_likelis_rescaled <- t(object[['Test_Data']][['Averaged_Likelis_Post_Thresh_Test']])
  num_samples <- dim(averaged_likelis_rescaled)[1]
  num_points <- dim(averaged_likelis_rescaled)[2]
  thetas <- c()
  for (i in 1:num_samples) {
    curr_sample <- exp(averaged_likelis_rescaled[i,])
    curr_lrf_curve <- curr_sample / max(curr_sample)
    curr_curve <- suppressWarnings(eta*(1 + epsilon + cos(2*pi*((1:num_points)/num_points - which(curr_lrf_curve == max(curr_lrf_curve))/num_points))))

    lrf_curve_spline <- stats::predict(splines::periodicSpline(1:num_points,curr_lrf_curve, period = num_points),seq(1,num_points,length.out = 1000))
    curve_spline <- stats::predict(splines::periodicSpline(1:num_points,curr_curve, period = num_points),seq(1,num_points,length.out = 1000))

    thetas[i] <- sum(lrf_curve_spline$y > curve_spline$y) / length(lrf_curve_spline$y)
  }
  object[['Test_Data']][['Thetas_Test']] <- thetas
  return(object)
}

calc_flat_theta_contrib_test <- function(object) {
  epsilon <- object[['Train_Data']][['epsilon']]
  eta <- object[['Train_Data']][['eta']]
  averaged_likelis_rescaled <- t(object[['Test_Data']][['Averaged_Likelis_Post_Thresh_Test']])
  num_samples <- dim(averaged_likelis_rescaled)[1]
  num_points <- dim(averaged_likelis_rescaled)[2]
  flat_contribution <- c()
  theta <- c()
  for (i in 1:num_samples) {
    ind_ts <- exp(averaged_likelis_rescaled[i,])
    ind_ts <- shift_ts(ind_ts, round(num_points/2))
    ind_lrf_curve <- ind_ts / max(ind_ts)

    curr_curve <- suppressWarnings(eta*(1 + epsilon + cos(2*pi*((1:num_points)/num_points - which(ind_lrf_curve == max(ind_lrf_curve))/num_points))))
    lrf_curve_spline <- stats::predict(splines::periodicSpline(1:num_points,ind_lrf_curve, period = num_points),seq(1,num_points,length.out = 1000))
    curve_spline <- stats::predict(splines::periodicSpline(1:num_points,curr_curve, period = num_points),seq(1,num_points,length.out = 1000))

    theta_index <- which(lrf_curve_spline$y > curve_spline$y)
    flat_regions <- which(abs(diff(lrf_curve_spline$y)) < 1e-4)
    flat_contribution[i] <- sum(theta_index %in% flat_regions) / length(theta_index) * 100
  }
  flat_contribution_df <- data.frame(flat_contributions = flat_contribution, thetas = object[['Test_Data']][['Thetas_Test']])
  object[['Test_Data']][['Flat_Contrib_to_Theta_df']] <- flat_contribution_df
  return(object)
}

second_peaks_fun_test <- function(object, minpeakheight = -Inf, minpeakdistance = 1, nups = 1, ndowns = 0, threshold = 0, npeaks = 2) {
  likelis_array <- object[['Test_Data']][['Test_Likelihood_Array']]
  logthresh <- object[['Test_Data']][['LogThresh_Test']]
  # Get contribution of flat regions to theta
  flat_contributions_df <- object[['Test_Data']][['Flat_Contrib_to_Theta_df']]
  # Looking at averaged Likelihoods after truncation
  averaged_likelis <- object[['Test_Data']][['Averaged_Likelis_Post_Thresh_Test']]
  actual_time <- as.numeric(object[['Metadata']][['Test']][['Time']]) %% 24

  npoints <- dim(averaged_likelis)[1]
  peaks_list <- apply(averaged_likelis, 2, pracma::findpeaks, nups = nups, ndowns = ndowns,
                      sortstr = TRUE, minpeakheight = minpeakheight, minpeakdistance = minpeakdistance, threshold = threshold, npeaks = npeaks, simplify = FALSE)
  npeaks_list <- purrr::map(peaks_list, ~nrow(.x))
  npeaks_list[sapply(npeaks_list, is.null)] <- NA
  npeaks <- unlist(npeaks_list)
  log_values <- purrr::map(peaks_list, ~.x[,1])
  log_values[sapply(log_values, is.null)] <- NA
  times <- purrr::map(peaks_list, ~.x[,2])
  times[sapply(times, is.null)] <- NA
  times_hours <- purrr::map(times, ~.x / npoints * 24)
  times_1st_peak <- (purrr::map_dbl(times_hours, ~.x[1]) + object[['Metadata']][['Train']][['min_T_mod24']]) %% 24
  times_2nd_peak <- (purrr::map_dbl(times_hours, ~.x[2]) + object[['Metadata']][['Train']][['min_T_mod24']]) %% 24
  log_likeli_1st_peak <- purrr::map_dbl(log_values, ~.x[1])
  log_likeli_2nd_peak <- purrr::map_dbl(log_values, ~.x[2])
  peaks_df <- data.frame(npeaks = npeaks, time_1st_peak = round(times_1st_peak,2), time_2nd_peak = round(times_2nd_peak,2),
                         max_1st_peak = round(log_likeli_1st_peak,3), max_2nd_peak = round(log_likeli_2nd_peak,3))
  peaks_df <- peaks_df %>% dplyr::mutate(times_diff = time_1st_peak - time_2nd_peak, max_diff = max_1st_peak - max_2nd_peak)

  percent_flat <- function(likelis, thresh) {
    out <- round(sum(likelis == thresh) / npoints * 100,1)
    return(out)
  }

  peaks_df$PercFlat <- apply(averaged_likelis,2,percent_flat, logthresh)
  peaks_df$FlatContrib <- flat_contributions_df$flat_contributions
  peaks_df$Theta <- flat_contributions_df$thetas

  # Looking at original projections before truncation
  local_max_vals <- apply(likelis_array, c(2,3), max)
  peaks_df <- peaks_df %>% dplyr::mutate(local_smallest_peak = round(apply(local_max_vals,1,min),3), local_biggest_peak = round(apply(local_max_vals,1,max),3),
                                         local_mean = round(apply(local_max_vals,1,mean),3), local_sd = round(apply(local_max_vals,1,sd),3))

  local_times <- (apply(likelis_array, c(2,3), which.max) / npoints * 24 + object[['Metadata']][['Train']][['min_T_mod24']]) %% 24
  time_weights <- t(apply(local_max_vals, 1, function(x)(x-min(x))/(max(x)-min(x))))

  # Weighted average time prediction using max likelihoods values
  weighted_times_vec <- apply(sweep(local_times*time_weights, 1, apply(time_weights,1,sum), '/'),1,sum)

  local_times_rad <- local_times / 24 * 2 * pi
  mean_local_times <- (24 + suppressWarnings(apply(local_times_rad, 1, circular::mean.circular)) * 24 / 2 / pi) %% 24
  sd_local_times <- suppressWarnings(apply(local_times_rad, 1, circular::meandeviation)) * 24 / 2 / pi
  peaks_df <- peaks_df %>% dplyr::mutate(mean_time_pred = round(mean_local_times,2), sd_time_pred = round(sd_local_times,2),
                                         min_time_pred = round(apply(local_times,1,min),2), max_time_pred = round(apply(local_times,1,max),2),
                                         weighted_mean_time_pred = round(weighted_times_vec,2)) %>%
    dplyr::mutate(time_pred_diff = time_1st_peak - weighted_mean_time_pred)

  # In case no peaks are above the threshold, to avoid having NA for predicted time, I use weighted mean of time predictions from local peaks without threshold
  peaks_df$time_1st_peak <- ifelse(is.na(peaks_df$time_1st_peak), peaks_df$weighted_mean_time_pred,peaks_df$time_1st_peak)
  peaks_df$Actual_Time <- actual_time
  peaks_df$Pred_Error <- circular_difference(peaks_df$time_1st_peak, peaks_df$Actual_Time)

  peaks_df <- peaks_df %>% dplyr::mutate(Group_1 = object[['Metadata']][['Test']][['Group_1']],
                                         Group_2 = object[['Metadata']][['Test']][['Group_2']],
                                         Group_3 = object[['Metadata']][['Test']][['Group_3']],
                                         Replicate = object[['Metadata']][['Test']][['Replicate']])

  object[['Test_Data']][['Results_df']] <- peaks_df
  return(object)
}

shift_outlier_times <- function(object, minT = 8, maxT = 20) {
  data <- object[['Test_Data']][['Results_df']] %>%
    dplyr::mutate(flagged_1st = if_else(time_1st_peak < minT | time_1st_peak > maxT, TRUE, FALSE), .after = time_1st_peak) %>%
    dplyr::mutate(flagged_2nd = if_else(time_2nd_peak < minT | time_2nd_peak > maxT, TRUE, FALSE), .after = time_2nd_peak) %>%
    dplyr::mutate(Corrected_Time = if_else(flagged_1st, if_else(flagged_2nd, time_1st_peak, time_2nd_peak), time_1st_peak),
                  Was_Shifted = if_else(flagged_1st, if_else(flagged_2nd, 0, 1), 0))

  object[['Test_Data']][['Results_df']] <- data

  message('Number of Time Predictions that were shifted to the defined window if applicable: ', sum(object[['Test_Data']][['Results_df']]$Was_Shifted, na.rm = TRUE))
  return(object)
}
