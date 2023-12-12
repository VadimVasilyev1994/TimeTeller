make_data_object <- function(exp_matrix, genes, group_1, group_2, group_3, time, replicate, mat_normalised) {
  timeteller_list <- list()
  if (missing(group_1)) {
    message('Missing group information. Please provide high level group information or use intergene normalisation.')
    group_1 <- rep(NA, dim(exp_matrix)[2])
  }
  if (missing(group_2)) {group_2 <- rep(NA, dim(exp_matrix)[2])}
  if (missing(group_3)) {group_3 <- rep(NA, dim(exp_matrix)[2])}
  if (missing(replicate)) {replicate <- rep(NA, dim(exp_matrix)[2])}
  if (missing(time)) {stop("Model can't be trained without training times. Please provide time vector")}
  ll <- list(group_1,group_2,group_3,time,replicate)
  if (!all(sapply(ll,length)==length(ll[[1]]))) {stop("Supplied metadata is not all of equal length. Please check inputs")}
  if (!all(genes %in% rownames(exp_matrix))) {stop("Not all training genes found in the supplied data. Please check inputs")}
  if (mat_normalised) {
    timeteller_list[['Full_Original_Data_Raw']] <- 'Not Provided'
    timeteller_list[['Full_Original_Data']] <- as.matrix(exp_matrix)
  } else {
    timeteller_list[['Full_Original_Data_Raw']] <- as.matrix(exp_matrix)
    timeteller_list[['Full_Original_Data']] <- prepare_raw_counts(exp_matrix)
  }
  exp_df <- data.table::transpose(as.data.frame(exp_matrix[genes,]))
  colnames(exp_df) <- paste('Gene_', genes, sep='')
  full_df <- exp_df %>% dplyr::mutate(Group = as.character(group_1), Group_2 = as.character(group_2), Group_3 = as.character(group_3), Time = as.factor(time), Replicate = as.character(replicate))
  full_df <- full_df %>% dplyr::mutate_at(c('Group','Group_2','Group_3','Replicate'), ~tidyr::replace_na(.,""))
  cat('Please check the information provided:\n')
  cat(paste('# Number of genes used:',length(genes), ' (',paste(genes,collapse = ' '),')\n'))
  cat(paste('# Number of Group_1 levels:',length(unique(group_1)), ' (',paste(unique(group_1),collapse = ' '),')\n'))
  cat(paste('# Number of Group_2 levels:',length(unique(group_2)), ' (',paste(unique(group_2),collapse = ' '),')\n'))
  cat(paste('# Number of Group_2 levels:',length(unique(group_3)), ' (',paste(unique(group_3),collapse = ' '),')\n'))
  cat(paste('# Number of Replicate levels:',length(unique(replicate)), ' (',paste(unique(replicate),collapse = ' '),')\n'))
  cat(paste('# Number of unique time points Used for training:',length(unique(time)), ' (',paste(unique(time),collapse = ' '),')\n'))
  timeteller_list[['Train']][['Data']] <- full_df
  timeteller_list[['Train']][['Train_Exp_Matrix']] <- exp_matrix[genes,]
  timeteller_list[['Metadata']][['Train']][['Genes_Used']] <- genes
  timeteller_list[['Metadata']][['Train']][['Group_1']] <- group_1
  timeteller_list[['Metadata']][['Train']][['Group_2']] <- group_2
  timeteller_list[['Metadata']][['Train']][['Group_3']] <- group_3
  timeteller_list[['Metadata']][['Train']][['Time']] <- time
  timeteller_list[['Metadata']][['Train']][['Replicate']] <- replicate
  return(timeteller_list)
}

deal_with_replicates <- function(object, treat_independently = TRUE) {
  # Start with averaging the replicates if the user decided to
  data <- object[['Train']][['Data']]
  if (treat_independently) {
    cat('Replicates (if there are any) will be treated as independent observations for the training model\n')
    object[['Train']][['Data']] <- data
    return(object)
  } else {
    cat('Replicates for the same groups and time will be averaged before training the model\n')
    counts_df <- as.data.frame(table(data$Group,data$Group_2,data$Group_3,data$Time)) %>% dplyr::filter(Freq > 1)
    final_df <- data
    if (dim(counts_df)[1] > 0) {
      for (i in 1:dim(counts_df)[1]) {
        curr_group <- counts_df[i,1]
        curr_group_2 <- counts_df[i,2]
        curr_group_3 <- counts_df[i,3]
        curr_time <- counts_df[i,4]
        curr_df <- final_df %>% dplyr::filter(Group == curr_group, Group_2 == curr_group_2, Group_3 == curr_group_3, Time == curr_time) %>% dplyr::select(starts_with('Gene_'))
        averaged_df <- curr_df %>% dplyr::summarise(across(.cols = everything(), mean)) %>% dplyr::mutate(Group = curr_group, Group_2 = curr_group_2, Group_3 = curr_group_3, Time = curr_time, Replicate = '')
        final_df <- final_df %>% dplyr::rows_delete(final_df %>% dplyr::filter(Group == curr_group, Group_2 == curr_group_2, Group_3 == curr_group_3, Time == curr_time)) %>% dplyr::rows_insert(averaged_df, conflict = 'ignore')
      }
      cat('There were ', dim(counts_df)[1],' ','groups that were averaged\n')
      object[['Train']][['Data']] <- final_df
    }
    else {
      cat('No replicates were found so no averaging was done. Please check inputs if replicates expected\n')
      object[['Train']][['Data']] <- final_df
    }
    return(object)
  }
}

allocate_timeseries_ind <- function(object, combine_for_norm = FALSE) {
  data <- object[['Train']][['Data']]
  data$Time_mod24 <- as.numeric(as.character(data$Time)) %% 24

  if (combine_for_norm == TRUE) {
    data <- data %>% dplyr::group_by(Group, Group_2, Group_3) %>% dplyr::mutate(timeseries_name = factor(paste0(as.character(Group), as.character(Group_2), as.character(Group_3)))) %>%
      dplyr::mutate(timeseries_ind = as.numeric(timeseries_name))
  } else {
    data <- data %>% dplyr::group_by(Group, Group_2, Group_3, Replicate) %>% dplyr::mutate(timeseries_name = factor(paste0(as.character(Group), as.character(Group_2), as.character(Group_3), as.character(Replicate)))) %>%
      dplyr::mutate(timeseries_ind = as.numeric(timeseries_name))
  }

  object[['Train']][['Data']] <- data
  return(object)
}

normalised_df <- function(object, method = 'intergene', grouping_vars = c('Group')) {
  data <- object[['Train']][['Data']]
  genes <- colnames(data %>% ungroup() %>% dplyr::select(starts_with('Gene_')))
  object[['Normalisation_choice']] <- method

  if(method == 'timecourse') {
    cat('Using timecourse normalisation\n')
    timeseries_norm_df <- data %>% dplyr::group_by(timeseries_ind) %>% dplyr::mutate(across(all_of(genes), .fns = list(normalised = ~(.x - mean(.x))/sd(.x)))) %>%
      dplyr::rename_at( vars( contains( "_normalised") ), list( ~paste("normalised", gsub("_normalised", "", .), sep = "_") ) ) %>% dplyr::ungroup()
    timeseries_norm_df <- timeseries_norm_df %>% dplyr::mutate(time_group = factor(paste('Time_', Time_mod24, sep = ''), levels = c(paste('Time_', sort(unique(timeseries_norm_df$Time_mod24)), sep = '')))) %>%
      dplyr::arrange(Time_mod24)
    object[['Train']][['Normalised_Data']] <- timeseries_norm_df
    object[['Train']][['Normalised_Train_Exp_Data']] <- timeseries_norm_df %>% dplyr::ungroup() %>% dplyr::select(matches("normalised")) %>%
      data.table::transpose() %>% as.matrix()
    return(object)
  }

  else if(method == 'intergene'){
    cat('Using intergene normalisation\n')
    intergene_norm_df <- data %>% dplyr::ungroup() %>% rowwise() %>% dplyr::mutate(mean = mean(c_across(genes)), sd = sd(c_across(genes))) %>%
      dplyr::ungroup() %>% dplyr::mutate(across(all_of(genes), .fns = list(normalised = ~(.x - mean)/sd))) %>%
      dplyr::rename_at( vars( contains( "_normalised") ), list( ~paste("normalised", gsub("_normalised", "", .), sep = "_") ) )
    intergene_norm_df <- intergene_norm_df %>% dplyr::mutate(time_group = factor(paste('Time_', Time_mod24, sep = ''), levels = c(paste('Time_', sort(unique(intergene_norm_df$Time_mod24)), sep = '')))) %>%
      dplyr::arrange(Time_mod24)
    object[['Train']][['Normalised_Data']] <- intergene_norm_df
    object[['Train']][['Normalised_Train_Exp_Data']] <- intergene_norm_df %>% dplyr::ungroup() %>% dplyr::select(matches("normalised")) %>%
      data.table::transpose() %>% as.matrix()
    return(object)
  }

  else if(method == 'clr'){
    cat('Using clr normalisation\n')
    clr_norm_df <- data %>% dplyr::ungroup() %>% rowwise() %>% dplyr::mutate(geom_mean = exp(mean(log(c_across(genes))))) %>%
      dplyr::ungroup() %>% dplyr::mutate(across(all_of(genes), .fns = list(normalised = ~ (log(.x) - log(geom_mean))))) %>%
      dplyr::rename_at( vars( contains( "_normalised") ), list( ~paste("normalised", gsub("_normalised", "", .), sep = "_") ) )
    clr_norm_df <- clr_norm_df %>% dplyr::mutate(time_group = factor(paste('Time_', Time_mod24, sep = ''), levels = c(paste('Time_', sort(unique(clr_norm_df$Time_mod24)), sep = '')))) %>%
      dplyr::arrange(Time_mod24)
    object[['Train']][['Normalised_Data']] <- clr_norm_df
    object[['Train']][['Normalised_Train_Exp_Data']] <- clr_norm_df %>% dplyr::ungroup() %>% dplyr::select(matches("normalised")) %>%
      data.table::transpose() %>% as.matrix()
    return(object)
  }

  else if(method == 'timecourse_matched'){
    cat('Using timecourse-matched normalisation. Pay attention to the group names used when testing\n')
    timeseries_norm_df <- data %>% dplyr::group_by(timeseries_ind) %>% dplyr::mutate(across(all_of(genes), .fns = list(normalised = ~(.x - mean(.x))/sd(.x)))) %>%
      dplyr::rename_at( vars( contains( "_normalised") ), list( ~paste("normalised", gsub("_normalised", "", .), sep = "_") ) )
    timeseries_norm_df <- timeseries_norm_df %>% dplyr::mutate(time_group = factor(paste('Time_', Time_mod24, sep = ''), levels = c(paste('Time_', sort(unique(timeseries_norm_df$Time_mod24)), sep = '')))) %>%
      dplyr::arrange(Time_mod24)
    object[['Train']][['Normalised_Data']] <- timeseries_norm_df
    object[['Train']][['Normalised_Train_Exp_Data']] <- timeseries_norm_df %>% dplyr::ungroup() %>% dplyr::select(matches("normalised")) %>%
      data.table::transpose() %>% as.matrix()

    group_statistics <- data %>% dplyr::group_by(across(all_of(grouping_vars))) %>% dplyr::select(all_of(genes)) %>%
      dplyr::summarise_all(list(Mean = mean,SD = sd)) %>%
      dplyr::rename_at( vars( contains( "Gene_") ), list( ~paste("", gsub("Gene_", "", .), sep = "") ) ) %>%
      tidyr::unite('timeseries_matched_name', all_of(grouping_vars), remove = FALSE, sep = '|') %>%
      tibble::column_to_rownames(var = 'timeseries_matched_name')
    group_metrics_list <- sapply(c("Mean", "SD"), function(x) group_statistics[endsWith(names(group_statistics),x)], simplify = FALSE)
    object[['Train']][['Gene_Per_Group_Info']] <- group_metrics_list
    return(object)
  }

}

get_projections <- function(object, num_PC) {
  data <- object[['Train']][['Normalised_Data']]
  object[['PC_Num']] <- num_PC
  group_list <- data %>% dplyr::ungroup() %>% dplyr::select(starts_with('normalised_')) %>% split.data.frame(data$time_group)
  group_list <- lapply(group_list, as.matrix)
  group_list <- lapply(group_list, base::t)
  object[['Projections']][['Training_Exp_Per_Time_Point']] <- group_list
  pr_comp_list <- purrr::map(group_list, as_mapper(~ base::t(svd(.x)$u[,1:num_PC])))
  # pr_comp_list <- purrr::map(group_list, as_mapper(~ base::t(svd(sweep(.x, 1, apply(.x, 1, mean), '-'))$u[,1:num_PC])))
  var_explained_list <- purrr::map(group_list, as_mapper(~ cumsum(svd(.x)$d^2 / sum(svd(.x)$d^2) * 100)))
  # var_explained_list <- purrr::map(group_list, as_mapper(~ cumsum(svd(sweep(.x, 1, apply(.x, 1, mean), '-'))$d^2 / sum(svd(.x)$d^2) * 100)))
  object[['Projections']][['SVD_Per_Time_Point']] <- pr_comp_list
  object[['Projections']][['SVD_Per_Time_Point_Var_Explained']] <- var_explained_list
  all_projections <- list()
  for (j in seq_along(names(pr_comp_list))) {
    local_proj_list <- list()
    for (i in seq_along(names(group_list))) {
      local_proj_list[[names(group_list)[i]]] <- pr_comp_list[[j]] %*% group_list[[i]]
    }
    pc_vec <- paste('PC', seq(1:num_PC), sep = '')
    df <-  as.data.frame(t(do.call(cbind, local_proj_list)))
    colnames(df) <- pc_vec
    df <- df %>% dplyr::mutate(sample_times = data$Time_mod24, Group = data$Group, Group_2 = data$Group_2, Group_3 = data$Group_3)
    all_projections[[names(pr_comp_list)[j]]] <- df
  }
  object[['Projections']][['All_Projections']] <- all_projections
  return(object)
}

get_mvn_original_data <- function(object, cov_estimate = 'normal', alpha_par = 0.7) {
  if (cov_estimate == 'normal') {cat('Using standard Covariance and Location estimation\n')}
  if (cov_estimate == 'robust') {cat('Using Robust Covariance and Location Estimation (Fast MCD)\n')}
  object[['Projections']][['Cov_Method']] <- cov_estimate
  num_PC <- object[['PC_Num']]
  projection_list <- object[['Projections']][['All_Projections']]
  likeli_array_original <- base::array(data = NA, dim = c(num_PC + num_PC^2, length(names(projection_list)), length(names(projection_list))))
  dimnames(likeli_array_original) <- list(c(paste0('mean_PC',1:num_PC),paste0('sigma',1:num_PC^2)), names(projection_list), names(projection_list))
  for (i in 1:length(names(projection_list))) {
    curr_projection <- projection_list[[i]]
    data.split <- split(curr_projection[, 1:num_PC], curr_projection$sample_times)
    if (cov_estimate == 'normal') {
      data.mvnorm <- lapply(data.split, function(x) Rfast::mvnorm.mle(as.matrix(x)))
      mat <- purrr::map(data.mvnorm, extract_elements_func)
      mat_final <- base::matrix(unlist(mat), nrow = num_PC + num_PC^2)
      likeli_array_original[,,i] <- mat_final
    }
    if (cov_estimate == 'robust') {
      check_pkg("rrcov")
      data.mvnorm <- lapply(data.split, function(x) rrcov::CovMcd(as.matrix(x), alpha = alpha_par, nsamp = 'deterministic'))
      mat <- purrr::map(data.mvnorm, extract_elements_func_robust)
      mat_final <- base::matrix(unlist(mat), nrow = num_PC + num_PC^2)
      likeli_array_original[,,i] <- mat_final
    }
  }

  object[['Projections']][['Fitted_MVN_Original']] <- likeli_array_original
  return(object)
}

get_mvn_interpolated <- function(object, num_interp_points = 144, interp_method = 'perpchip', cov_path = 'spline') {
  likelihood_array <- object[['Projections']][['Fitted_MVN_Original']]
  numPC <- object[['PC_Num']]
  dims_array <- dim(likelihood_array)
  interpolated_array <- base::array(data = NA, dim = c(dims_array[1], num_interp_points, dims_array[3]))
  times <- as.numeric(sub(".*Time_", "", dimnames(likelihood_array)[[2]]))
  times_periodic <- c(times, times[1]+24)
  object[['Metadata']][['Train']][['min_T_mod24']] <- min(times_periodic)
  object[['Metadata']][['Train']][['max_T_mod24']] <- max(times_periodic)
  if (cov_path == 'spline') {
    for (j in 1:dims_array[3]) {
      for (i in 1:dims_array[1]) {
        ts_to_interpolate <- likelihood_array[i,,j]
        ts_to_interpolate_periodic <- c(ts_to_interpolate, ts_to_interpolate[1])

        if (interp_method == 'perpchip') {
          interpolated_array[i,,j] <- perpchip(times_periodic, ts_to_interpolate_periodic, seq(min(times_periodic), max(times_periodic), length.out = num_interp_points))
        }
        if (interp_method == 'standard') {
          interpolated_array[i,,j] <- stats::spline(times_periodic, ts_to_interpolate_periodic, n = num_interp_points, method = "periodic")$y
        }

      }
    }
    dimnames(interpolated_array) <- list(dimnames(likelihood_array)[[1]], paste0('Time_',round(seq(min(times_periodic), max(times_periodic), length.out = num_interp_points),2)), dimnames(likelihood_array)[[3]])
    object[['Projections']][['Fitted_MVN_Interpolated']] <- interpolated_array
    return(object)
  }


  if (cov_path == 'fisherrao') {
    for (j in 1:dims_array[3]) {

      num_points <- num_interp_points + dims_array[3]
      times_num <- as.numeric(substring(colnames(likelihood_array), first = 6))
      interpolation_points <- c()
      for (i in 1:length(times_num)) {
        interpolation_points[i] <- length(seq(0, 1, length.out = floor(((abs(circular_difference(times_num[i],times_num[ifelse(i+1>length(times_num),1,i+1)])))*60) / (24*60/num_points))))
      }
      interp_res <- list()
      for (i in 1:length(times_num)) {
        cov_mat_1 <- matrix(likelihood_array[(numPC+1):(numPC+numPC^2),i,j], nrow = 3)
        cov_mat_2 <- matrix(likelihood_array[(numPC+1):(numPC+numPC^2),ifelse(i+1>length(times_num),1,i+1),j], nrow = 3)
        interp_points <- seq(0, 1, length.out = interpolation_points[i])

        mat_list <- matrix(NA, nrow = numPC^2, ncol = length(interp_points))
        for (k in 1:length(interp_points)) {
          first_term <- pracma::sqrtm(cov_mat_1)$B
          second_term <- solve(pracma::sqrtm(cov_mat_1)$B) %*% cov_mat_2 %*% solve(pracma::sqrtm(cov_mat_1)$B)
          second_term <- expm(logm(second_term) * interp_points[k])
          third_term <- pracma::sqrtm(cov_mat_1)$B

          interp_mat <- first_term %*% second_term %*% third_term
          mat_list[,k] <- as.vector(interp_mat)
        }
        mat_names <- paste0('Time_', round(seq(times_num[i], ifelse(i+1>length(times_num),times_num[1]+24,times_num[i+1]), length.out = interpolation_points[i]),2))
        colnames(mat_list) <- mat_names
        if (i>1) {mat_list <- mat_list[,-1]}
        interp_res[[i]] <- mat_list
      }

      check <- do.call(cbind,interp_res)
      check <- check[, !duplicated(colnames(check))]
      interpolated_array <- interpolated_array[ , 1:dim(check)[2], ]
      interpolated_array[(numPC+1):(numPC+numPC^2), ,j] <- check


      for (i in 1:numPC) {
        ts_to_interpolate <- likelihood_array[i,,j]
        ts_to_interpolate_periodic <- c(ts_to_interpolate, ts_to_interpolate[1])

        if (interp_method == 'perpchip') {
          interpolated_array[i,,j] <- perpchip(times_periodic, ts_to_interpolate_periodic, seq(min(times_periodic), max(times_periodic), length.out = dim(check)[2]))
        }
        if (interp_method == 'standard') {
          interpolated_array[i,,j] <- stats::spline(times_periodic, ts_to_interpolate_periodic, n = dim(check)[2], method = "periodic")$y
        }

      }

      dimnames(interpolated_array)[[2]] <- colnames(check)

    }
    dimnames(interpolated_array) <- list(dimnames(likelihood_array)[[1]], dimnames(interpolated_array)[[2]], dimnames(likelihood_array)[[3]])


    object[['Projections']][['Fitted_MVN_Interpolated']] <- interpolated_array
    return(object)
  }
}

calc_train_likelis <- function(object) {
  svd_data <- object[['Projections']][['SVD_Per_Time_Point']]
  fitted_mvn_data <- object[['Projections']][['Fitted_MVN_Interpolated']]
  train_exp_data <- object[['Train']][['Normalised_Train_Exp_Data']]
  num_PC <- object[['PC_Num']]
  train_likelihood_array <- base::array(data = NA, dim = c(dim(fitted_mvn_data)[2], dim(train_exp_data)[2], dim(fitted_mvn_data)[3]))
  for (i in 1:length(names(svd_data))) {
    project_exp_mat <- svd_data[[i]] %*% train_exp_data
    for (ind_num in 1:dim(train_exp_data)[2]) {
      vec <- c()
      for(j in 1:dim(fitted_mvn_data)[2]) {
        curr_sigma <- matrix(fitted_mvn_data[(num_PC+1):(num_PC+num_PC^2),j,i], nrow = num_PC)
        curr_eig <- eigen(curr_sigma)$values
        if (any(curr_eig < 0)) {
          sigma_used <- nearPD(curr_sigma, base.matrix = TRUE, ensureSymmetry = TRUE, eig.tol = 1e-06, conv.tol = 1e-07, posd.tol = 1e-07)$mat
        } else {
          sigma_used <- curr_sigma
        }

        vec[j] <- mvtnorm::dmvnorm(project_exp_mat[,ind_num], mean = fitted_mvn_data[1:num_PC,j,i], sigma = sigma_used, checkSymmetry = FALSE)
      }
      train_likelihood_array[,ind_num,i] <- vec
    }
    Sys.sleep(0.01)
    cat("\rFinished", i, "of", length(names(svd_data)), "\n")
  }
  object[['Train_Data']][['Train_Likelihood_Array']] <- log(train_likelihood_array)
  return(object)
}

get_final_likelis_train <- function(object, log_thresh){
  likelis_array <- object[['Train_Data']][['Train_Likelihood_Array']]
  object[['Train_Data']][['LogThresh_Train']] <- log_thresh
  num_time_points <- dim(likelis_array)[3]
  for (i in 1:num_time_points) {
    likelis_array[,,i] <- pmax(likelis_array[,,i], log_thresh)
  }
  object[['Train_Data']][['Likelis_Post_Thresh_Train']] <- likelis_array
  final_averaged_likelis <- apply(likelis_array, c(1,2), mean)
  object[['Train_Data']][['Averaged_Likelis_Post_Thresh_Train']] <- final_averaged_likelis
  max_likelis <- apply(final_averaged_likelis,2,max)
  object[['Train_Data']][['Max_Likelis_Train']] <- max_likelis
  return(object)
}

theta_calc_train <- function(object, epsilon = 0.4, eta = 0.35) {
  averaged_likelis_rescaled <- t(object[['Train_Data']][['Averaged_Likelis_Post_Thresh_Train']])
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
  object[['Train_Data']][['Thetas_Train']] <- thetas
  object[['Train_Data']][['epsilon']] <- epsilon
  object[['Train_Data']][['eta']] <- eta
  return(object)
}

calc_flat_theta_contrib_train <- function(object) {
  averaged_likelis_rescaled <- t(object[['Train_Data']][['Averaged_Likelis_Post_Thresh_Train']])
  num_samples <- dim(averaged_likelis_rescaled)[1]
  num_points <- dim(averaged_likelis_rescaled)[2]
  epsilon <- object[['Train_Data']][['epsilon']]
  eta <- object[['Train_Data']][['eta']]
  flat_contribution <- c()
  theta <- c()
  for (i in 1:num_samples) {
    ind_ts <- exp(averaged_likelis_rescaled[i,])
    ind_ts <- shift_ts(ind_ts, round(num_points/2))
    ind_lrf_curve <- ind_ts / max(ind_ts)

    curr_curve <- suppressWarnings(eta*(1 + epsilon + cos(2*pi*((1:num_points)/num_points - which(ind_lrf_curve == max(ind_lrf_curve))/num_points))))
    lrf_curve_spline <- stats::predict(periodicSpline(1:num_points,ind_lrf_curve, period = num_points),seq(1,num_points,length.out = 1000))
    curve_spline <- stats::predict(periodicSpline(1:num_points,curr_curve, period = num_points),seq(1,num_points,length.out = 1000))

    theta_index <- which(lrf_curve_spline$y > curve_spline$y)
    flat_regions <- which(abs(diff(lrf_curve_spline$y)) < 1e-4)
    flat_contribution[i] <- sum(theta_index %in% flat_regions) / length(theta_index) * 100
  }
  flat_contribution_df <- data.frame(flat_contributions = flat_contribution, thetas = object[['Train_Data']][['Thetas_Train']])
  object[['Train_Data']][['Flat_Contrib_to_Theta_df']] <- flat_contribution_df
  return(object)
}

second_peaks_fun_train <- function(object, minpeakheight = -Inf, minpeakdistance = 1, nups = 1, ndowns = 0, threshold = 0, npeaks = 2) {
  likelis_array <- object[['Train_Data']][['Train_Likelihood_Array']]
  logthresh <- object[['Train_Data']][['LogThresh_Train']]
  # Get contribution of flat regions to theta
  flat_contributions_df <- object[['Train_Data']][['Flat_Contrib_to_Theta_df']]
  # Looking at averaged Likelihoods after truncation
  averaged_likelis <- object[['Train_Data']][['Averaged_Likelis_Post_Thresh_Train']]
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

  peaks_df$Actual_Time <- object$Train$Normalised_Data$Time_mod24
  peaks_df$Pred_Error <- circular_difference(peaks_df$time_1st_peak, peaks_df$Actual_Time)
  peaks_df$Group <- object$Train$Normalised_Data$Group
  peaks_df$Group_2 <- object$Train$Normalised_Data$Group_2
  peaks_df$Group_3 <- object$Train$Normalised_Data$Group_3
  peaks_df$Replicate <- object$Train$Normalised_Data$Replicate

  object[['Train_Data']][['Results_df']] <- peaks_df

  return(object)
}
