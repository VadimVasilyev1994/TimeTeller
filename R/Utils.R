perpchipend <- function(h1, h2, del1, del2) {
  d <- ((2 * h1 + h2) * del1 - h1 * del2)/(h1 + h2)
  if (sign(d) != sign(del1)) {
    d <- 0
  }
  else if ((sign(del1) != sign(del2)) && (abs(d) > abs(3 * del1))) {
    d <- 3 * del1
  }
  return(d)
}

perpchip_slopes <- function(h, delta) {
  n <- length(h) + 1
  d <- numeric(length(h))
  k <- which(sign(delta[1:(n - 2)]) * sign(delta[2:(n - 1)]) > 0) + 1
  w1 <- 2 * h[k] + h[k - 1]
  w2 <- h[k] + 2 * h[k - 1]
  d[k] <- (w1 + w2)/(w1/delta[k - 1] + w2/delta[k])
  d[1] <- perpchipend(h[1], h[2], delta[1], delta[2])
  d[n] <- perpchipend(h[n - 1], h[n - 2], delta[n - 1], delta[n - 2])
  return(d)
}

perpchip <- function(xi,yi,x) {
  stopifnot(is.numeric(xi), is.numeric(yi), is.numeric(x))
  if (!pracma::is.sorted(xi))
    stop("Argument 'xi' must be a sorted vector of real numbers.")
  n <- length(xi)
  if (length(yi) != n)
    stop("Arguments 'xi', 'yi' must be vectors of equal length.")
  if (n <= 2)
    stop("At least three points needed for cubic interpolation.")
  h <- diff(xi)
  delta <- diff(yi) / h
  x_used <- xi
  y_used <- yi

  # Processing data to make it periodic in end conditons
  yi <- c(yi[length(yi)-1],yi,yi[2])
  xi <- c(xi[1]-(xi[length(xi)]-xi[length(xi)-1]),xi,xi[length(xi)]+(xi[2]-xi[1]))
  deltai <- c(delta[length(delta)], delta, delta[1])
  hi <- diff(xi)

  d <- perpchip_slopes(hi, deltai)
  d <- d[2:(length(d)-1)]

  # Fitting to the original non-extended data now
  a <- (3 * delta - 2 * d[1:(n - 1)] - d[2:n])/h
  b <- (d[1:(n - 1)] - 2 * delta + d[2:n])/h^2
  k <- rep(1, length(x))
  for (j in 2:(n - 1)) {
    k[x_used[j] <= x] <- j
  }
  s <- x - x_used[k]
  v <- y_used[k] + s * (d[k] + s * (a[k] + s * b[k]))
  return(v)
}

circular_difference <- function(time_1, time_2) {
  time_1_rad <- time_1 * 2 * pi / 24
  time_2_rad <- time_2 * 2 * pi / 24
  abs_diff_rad <- atan2(sin(time_1_rad-time_2_rad), cos(time_1_rad-time_2_rad))
  diff_hours <- abs_diff_rad * 24 / 2 / pi
  return(diff_hours)
}

shift_ts <- function(ts, target_peak) {
  obs_num <- length(ts)
  max_ind <- which.max(ts)
  extended_ts <- c(ts,ts,ts)
  shift_dist <- target_peak - max_ind
  new_ts <- c()
  for (i in 1:obs_num) {
    new_ts[i] <- extended_ts[i+obs_num-shift_dist]
  }
  return(new_ts)
}

extract_elements_func <- function(data) {
  return(c(unname(data$mu), unname(c(data$sigma))))
}

extract_elements_func_robust <- function(data) {
  return(c(unname(data$center), unname(c(data$cov))))
}

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

deviation_cv_corrected <- function(list_cv) {
  pred_errors <- purrr::map(cv_list, as_mapper(~ .x$Test_Data$Results_df$Pred_Error - median(.x$Test_Data$Results_df$Pred_Error)))
  return(unlist(pred_errors))
}

volume_func <- function(curr_sigma, alpha = 0.95, dims = 3) {
  crit_val <- qchisq(p = alpha, df = dims, lower.tail=FALSE) ^ (dims/2)
  det_sqrt <- sqrt(det(curr_sigma))
  gamma_func <- dims * gamma(dims/2)
  volume <- (2 * pi^(dims/2)) / gamma_func * crit_val^(dims/2) * det_sqrt
  return(volume)
}

prepare_raw_counts <- function(exp_matrix) {
  check_pkg('edgeR', bioc = TRUE)
  obj_dge <- edgeR::DGEList(counts=as.matrix(exp_matrix))
  keep <- edgeR::filterByExpr(obj_dge)
  obj_dge <- obj_dge[keep, , keep.lib.sizes=FALSE]
  logCPM <- edgeR::cpm(obj_dge, log=TRUE, prior.count = 2)
  return(logCPM)
}

get_se_obj <- function(object) {
  check_pkg('SummarizedExperiment', bioc = TRUE)
  counts_train <- object$Full_Original_Data
  counts_test <- object$Test_Data$Full_Test_Data
  common_names <- intersect(rownames(counts_train), rownames(counts_test))
  meta_train <- object$Train_Data$Results_df %>% dplyr::select(c(Group, Group_2, Group_3, Replicate, time_1st_peak, Actual_Time, Theta)) %>%
    dplyr::mutate(Dataset = 'Train')
  meta_test <- object$Test_Data$Results_df %>% dplyr::select(c(Group_1, Group_2, Group_3, Replicate, time_1st_peak, Actual_Time, Theta)) %>%
    dplyr::mutate(Group_1 = as.character(Group_1)) %>% dplyr::rename(Group = Group_1) %>% dplyr::mutate(Dataset = 'Test')
  combined_counts <- cbind(counts_train[common_names, ], counts_test[common_names, ])
  combined_meta <- rbind(meta_train, meta_test) %>% dplyr::mutate(across(where(is.character), ~na_if(., '')))
  SEobj <- SummarizedExperiment::SummarizedExperiment(assays = list(normalised_counts = combined_counts),
                                colData = combined_meta)
  return(SEobj)
}

get_tt_res <- function(object, train_or_test = 'test') {
  if (train_or_test == 'test') {
    return(object$Test_Data$Results_df)
  } else if (train_or_test == 'train') {
    return(object$Train_Data$Results_df)
  }
}
