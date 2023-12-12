#' Visualise gene expression across groups and replicates
#'
#' Quick exploratory plot of temporal gene expression across groups and replicates for the training data.
#'
#' @param object list containing circadianTT training model following \code{train_model}
#' @param gene gene of interest for visualisation
#' @param group1 groups of interest for plotting. If not supplied, all levels from \code{Group_1} will be plotted
#' @param group2 groups of interest for plotting. If not supplied, all levels from \code{Group_2} will be plotted
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returned is the object of class \code{ggplot}
#' @export
#'

plot_reps <- function(object, gene, group1, group2) {
  group_1 <- object[['Metadata']][['Train']][['Group_1']]
  group_2 <- object[['Metadata']][['Train']][['Group_2']]
  replicate <- object[['Metadata']][['Train']][['Replicate']]
  if(missing(group1)) {group1 <- as.character(unique(group_1))}
  if(missing(group2)) {group2 <- as.character(unique(group_2))}
  time <- object[['Metadata']][['Train']][['Time']]
  data <- data.frame(object$Full_Original_Data[gene,])
  gene_names <- paste0('Gene_', gene)
  colnames(data) <- gene_names
  data <- data %>%
    dplyr::mutate(Group = as.character(group_1), Group_2 = as.character(group_2), Time = factor(time, levels = sort(unique(time))), Replicate = as.character(replicate))
  pivoted_df <- pivot_longer(data, cols = all_of(gene_names), names_to = "Gene", values_to = "Expression")
  pivoted_df <- pivoted_df %>% dplyr::filter(Group %in% group1, Group_2 %in% group2) %>% dplyr::mutate_all(~replace_na(.,"Not Provided"))
  gg_gene <- ggplot(data = pivoted_df, mapping = aes(x = Time, y = Expression, group = Replicate)) + geom_point(aes(color = Replicate)) +
    ggplot2::facet_grid(rows = vars(Group), cols = vars(Group_2), scales = 'fixed') + geom_line(aes(color = Replicate)) + labs(title = "Genewise expression across groups") +
    theme(plot.title = element_text(size = 13, face = "bold", color = "black"))
  return(gg_gene)
}

#' Visualise multiple gene expression
#'
#' Quick exploratory plot of temporal gene expression across groups. Recommended use when there are no replicates for clarity of visualisation
#'
#' @param object list containing circadianTT training model following \code{train_model}
#' @param genes genes of interest for visualisation
#' @param group1 groups of interest for plotting. If not supplied, all levels from \code{Group_1} will be plotted
#' @param group2 groups of interest for plotting. If not supplied, all levels from \code{Group_2} will be plotted
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returned is the object of class \code{ggplot}
#' @export
#'

plot_genes <- function(object, genes, group1, group2) {
  group_1 <- object[['Metadata']][['Train']][['Group_1']]
  group_2 <- object[['Metadata']][['Train']][['Group_2']]
  replicate <- object[['Metadata']][['Train']][['Replicate']]
  if(missing(group1)) {group1 <- as.character(unique(group_1))}
  if(missing(group2)) {group2 <- as.character(unique(group_2))}
  time <- object[['Metadata']][['Train']][['Time']]
  data <- data.frame(t(object$Full_Original_Data[genes,]))
  gene_names <- paste0('Gene_', genes)
  colnames(data) <- gene_names
  data <- data %>%
    dplyr::mutate(Group = as.character(group_1), Group_2 = as.character(group_2), Time = factor(time, levels = sort(unique(time))), Replicate = as.character(replicate))
  pivoted_df <- tidyr::pivot_longer(data, cols = all_of(gene_names), names_to = "Gene", values_to = "Expression")
  group_df <- pivoted_df %>% dplyr::filter(Group %in% group1, Group_2 %in% group2) %>% dplyr::mutate_all(~replace_na(.,"Not Provided"))
  gg_group <- ggplot(data = group_df, mapping = aes(x = Time, y = Expression, group = Gene)) + geom_point(aes(color = Gene)) +
    facet_grid(rows = vars(Group), cols = vars(Group_2)) + geom_line(aes(color = Gene)) + labs(title = "Groupwise normalised expression for the selected genes") +
    theme(plot.title = element_text(size = 12, face = "bold", color = "black"))
  return(gg_group)

}

#' Visualise the training model
#'
#' Training data projected in the principal component space.
#'
#' @param object list containing circadianTT training model following \code{train_model}
#' @param selected_local_projection training time selected for the local projection
#' @param density should covariance matrix cloud be displayed. Default is FALSE
#' @param opacity opacity level if \code{density = TRUE}
#' @param sig_level confidence level when \code{density = TRUE} and is used to control the size of the ellipsoid. This uses \code{rgl::ellipse3d}
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returned is the \code{plot_ly} object
#' @export
#'

plot_3d_projection <- function(object, selected_local_projection, density = FALSE, opacity = 0.05, sig_level = 0.90) {
  cov_method <- object[['Projections']][['Cov_Method']]
  projection_name <- paste0('Time_',selected_local_projection)
  data <- object[['Projections']][['All_Projections']][[`projection_name`]]
  # Initialise the plot and plot our samples
  colfunc <- colorRampPalette(c("red","yellow","springgreen","royalblue"))
  my_colors <- colfunc(length(unique(data$sample_times)))
  title = 'Local Projection'

  pc_num <- object[['PC_Num']]
  local_proj_spline <- data.frame(t(object[['Projections']][['Fitted_MVN_Interpolated']][1:pc_num,,projection_name]))
  local_proj_spline$sample_times <- seq(object[['Metadata']][['Train']][['min_T_mod24']], object[['Metadata']][['Train']][['min_T_mod24']], length.out = dim(object[['Projections']][['Fitted_MVN_Interpolated']])[2])
  my_colors_spline <- colfunc(dim(local_proj_spline)[1])

  knots_data <- data.frame(t(object[['Projections']][['Fitted_MVN_Original']][1:pc_num,,projection_name]))

  fig <- plot_ly() %>%
    add_trace(data = data, x = ~PC1, y = ~PC2, z = ~PC3, color = ~data$sample_times, colors = my_colors,
              type = 'scatter3d',mode = 'markers',text = ~paste('Group:', Group, '<br>Sample Time:', sample_times), name = 'Train Data') %>%
    add_trace(x = ~knots_data$mean_PC1, y = ~knots_data$mean_PC2, z = ~knots_data$mean_PC3, color = I('gray45'),
              type = 'scatter3d',mode = 'markers', text = ~paste('Projection:', rownames(knots_data)),
              marker = list(size = 8, symbol = 'diamond'), name = 'Knots') %>%
    add_trace(x = ~local_proj_spline$mean_PC1, y = ~local_proj_spline$mean_PC2, z = ~local_proj_spline$mean_PC3,
              type = "scatter3d", mode = "lines", opacity = 1, color = I('gray45'),
              line = list(width = 6), hoverinfo='skip', showlegend = FALSE, name = 'Spline for Train Data') %>%
    colorbar(title = "Time (Hours)")
  if (density) {
    if (cov_method == 'normal') {
      # Generate local MVN plots
      data.split <- split(data[, 1:3], data$sample_times)
      data.mvnorm <- lapply(data.split, function(x) mvnorm.mle(as.matrix(x)))
      mvn_names <- names(data.mvnorm)
      for (i in 1:length(data.mvnorm)) {
        curr_proj_time <- mvn_names[i]
        curr_mean <- data.mvnorm[[curr_proj_time]]$mu
        curr_sigma <- data.mvnorm[[curr_proj_time]]$sigma
        curr_ellipse <- rgl::ellipse3d(curr_sigma, centre = curr_mean, level = sig_level)
        fig <- fig %>%
          add_trace(x = curr_ellipse$vb[1,], y = curr_ellipse$vb[2,], z = curr_ellipse$vb[3,], color = I(my_colors[i]), opacity = opacity,
                    type = 'scatter3d',mode = 'markers', hoverinfo='skip', showlegend = FALSE, inherit = FALSE)
      }
    }
    if (cov_method == 'robust') {
      # Generate local MVN plots
      data.split <- split(data[, 1:3], data$sample_times)
      data.mvnorm <- lapply(data.split, function(x) rrcov::CovMcd(as.matrix(x), alpha = 0.8))
      mvn_names <- names(data.mvnorm)
      for (i in 1:length(data.mvnorm)) {
        curr_proj_time <- mvn_names[i]
        curr_mean <- data.mvnorm[[curr_proj_time]]$center
        curr_sigma <- data.mvnorm[[curr_proj_time]]$cov
        curr_ellipse <- rgl::ellipse3d(curr_sigma, centre = curr_mean, level = sig_level)
        fig <- fig %>%
          add_trace(x = curr_ellipse$vb[1,], y = curr_ellipse$vb[2,], z = curr_ellipse$vb[3,], color = I(my_colors[i]), opacity = opacity,
                    type = 'scatter3d',mode = 'markers', hoverinfo='skip', showlegend = FALSE, inherit = FALSE)
      }
    }
  }
  fig <- fig %>% layout(title = paste(title,projection_name), scene = list(bgcolor = "#e5ecf6"))
  return(fig)

}

#' Visualise the test data
#'
#' Test data projected onto the training model.
#'
#' @param object list containing circadianTT training and test models following \code{train_model} and \code{test_model} respectively
#' @param selected_local_projection training time selected for the local projection
#' @param density should covariance matrix cloud be displayed. Default is FALSE
#' @param opacity opacity level if \code{density = TRUE}
#' @param sig_level confidence level when \code{density = TRUE} and is used to control the size of the ellipsoid. This uses \code{rgl::ellipse3d}
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returned is the \code{plot_ly} object
#' @export
#'

plot_3d_projection_with_test <- function(object, selected_local_projection, density = FALSE, opacity = 0.05, sig_level = 0.90) {
  projection_name <- paste0('Time_',selected_local_projection)

  test_data_projections <- data.frame(t(object[['Projections']][['SVD_Per_Time_Point']][[`projection_name`]] %*% object[['Test_Data']][['Normalised_Test_Exp_Data']]))
  colnames(test_data_projections) <- paste0('PC',1:object[['PC_Num']])

  data <- object[['Projections']][['All_Projections']][[`projection_name`]]
  # Initialise the plot and plot our samples
  colfunc <- colorRampPalette(c("red","yellow","springgreen","royalblue"))
  my_colors <- colfunc(length(unique(data$sample_times)))
  title = 'Local Projection'

  local_proj_ind <- which(names(object[['Projections']][['All_Projections']]) == projection_name)
  pc_num <- object[['PC_Num']]
  local_proj_spline <- data.frame(t(object[['Projections']][['Fitted_MVN_Interpolated']][1:pc_num,,local_proj_ind]))
  local_proj_spline$sample_times <- seq(min(data$sample_times), max(data$sample_times), length.out = dim(object[['Projections']][['Fitted_MVN_Interpolated']])[2])
  my_colors_spline <- colfunc(dim(local_proj_spline)[1])

  if (length(object[['Test_Data']][['Thetas_Test']] < 40)) {selected_size <- 5}
  else if (length(object[['Test_Data']][['Thetas_Test']] < 80)) {selected_size <- 4}
  else if (length(object[['Test_Data']][['Thetas_Test']] < 120)) {selected_size <- 3}
  else if (length(object[['Test_Data']][['Thetas_Test']] < 250)) {selected_size <- 1.5}
  else {selected_size <- 0.75}

  fig <- plot_ly() %>%
    add_trace(data = data, x = ~PC1, y = ~PC2, z = ~PC3, color = ~data$sample_times, colors = my_colors,
              type = 'scatter3d',mode = 'markers',text = ~paste('Group:', Group, '<br>Sample Time:', sample_times, '<br>Theta:', object[['Train_Data']][['Thetas_Train']]),
              name = 'Train Data') %>%
    add_trace(x = ~test_data_projections$PC1, y = ~test_data_projections$PC2, z = ~test_data_projections$PC3,
              type = 'scatter3d',mode = 'markers',
              text = ~paste('Theta:', object[['Test_Data']][['Thetas_Test']], '<br>Row_Num:', 1:length(object[['Test_Data']][['Thetas_Test']]),
                            '<br>Pred_Time:', object[['Test_Data']][['Results_df']]$time_1st_peak, '<br>Actual_Time:',object[['Test_Data']][['Results_df']]$Actual_Time,
                            '<br>Group_1:',object[['Metadata']][['Test']][['Group_1']], '<br>Group_2:',object[['Metadata']][['Test']][['Group_2']],
                            '<br>Group_3:',object[['Metadata']][['Test']][['Group_3']], '<br>Replicate:',object[['Metadata']][['Test']][['Replicate']]),
              marker = list(color = ~object[['Test_Data']][['Thetas_Test']], size = selected_size, symbol = 'square'), name = 'Test Data', inherit = FALSE) %>%
    add_trace(x = ~local_proj_spline$mean_PC1, y = ~local_proj_spline$mean_PC2, z = ~local_proj_spline$mean_PC3,
              type = "scatter3d", mode = "lines", opacity = 1, color = I('gray45'),
              line = list(width = 6) , name = 'Spline for Train Data') %>%
    colorbar(title = "Time (Hours)")
  if (density) {
    # Generate local MVN plots
    data.split <- split(data[, 1:3], data$sample_times)
    data.mvnorm <- lapply(data.split, function(x) mvnorm.mle(as.matrix(x)))
    mvn_names <- names(data.mvnorm)
    for (i in 1:length(data.mvnorm)) {
      curr_proj_time <- mvn_names[i]
      curr_mean <- data.mvnorm[[curr_proj_time]]$mu
      curr_sigma <- data.mvnorm[[curr_proj_time]]$sigma
      curr_ellipse <- rgl::ellipse3d(curr_sigma, centre = curr_mean, level = sig_level)
      if (density) {
        fig <- fig %>%
          add_trace(x = curr_ellipse$vb[1,], y = curr_ellipse$vb[2,], z = curr_ellipse$vb[3,], color = I(my_colors[i]), opacity = opacity,
                    type = 'scatter3d',mode = 'markers', hoverinfo='skip', showlegend = FALSE, inherit = FALSE)
      }
    }
  }
  fig <- fig %>% layout(title = paste(title,projection_name), scene = list(bgcolor = "#e5ecf6"))
  return(fig)

}

#' Visualise the the estimated temporal expression of test data
#'
#' Plots both known temporal expression of the training data and estimated temporal expression of the test data. This means that times used for the test data are the ones predicted by \code{circadianTT}
#'
#' @param object list containing circadianTT training and test models following \code{train_model} and \code{test_model} respectively
#' @param genes genes / features of interest used for plotting
#' @param theta_thresh threshold used for theta classification into 'Good' and 'Bad' clocks. This can be subjective and is used for visualisation and sanity check
#' @param xlim_l lower limit for x axis. Helps with visualisation when samples cluster in a particular part of the day (eg biopsy samples taken mostly during day time)
#' @param xlim_u upper limit for x axis
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returned is the \code{ggplot} object
#' @export
#'

exprs_vs_PredTime_plot <- function(object, genes, theta_thresh, xlim_l = 10, xlim_u = 20) {

  if(missing(genes)) genes <- object[['Metadata']][['Train']][['Genes_Used']]

  train_data <- as.data.frame(t(object[['Full_Original_Data']][genes,])) %>% dplyr::mutate(Time = object[['Metadata']][['Train']][['Time']] %% 24) %>%
    dplyr::mutate(Theta = object[['Train_Data']][['Thetas_Train']], Clock_Status = if_else(Theta > theta_thresh, 'Bad','Good')) %>%
    dplyr::mutate(Dataset = 'Train')

  if ('Corrected_Time' %in% colnames(object[['Test_Data']][['Results_df']])) {
    test_info <- object[['Test_Data']][['Results_df']] %>% dplyr::select(Corrected_Time, Theta) %>% dplyr::rename(Time = Corrected_Time)
  } else {
    test_info <- object[['Test_Data']][['Results_df']] %>% dplyr::select(time_1st_peak, Theta) %>% dplyr::rename(Time = time_1st_peak)
  }

  test_data <- as.data.frame(t(object[['Test_Data']][['Full_Test_Data']][genes,]))
  test_df <- cbind(test_info,test_data) %>% dplyr::mutate(Clock_Status = if_else(Theta > theta_thresh, 'Bad','Good')) %>%
    dplyr::mutate(Dataset = 'Test')

  combined_df <- rbind(train_data, test_df) %>% tidyr::pivot_longer(cols = -c(Time, Theta, Clock_Status, Dataset), names_to = 'Gene_Name', values_to = 'Expression')

  time_vs_exp_plot <- combined_df %>% ggplot2::ggplot(aes(x = Time, y = Expression, color = Clock_Status)) +
    geom_point(aes(shape = Dataset, size = Dataset)) + scale_size_manual(values=c(1,3)) +
    geom_smooth(data = combined_df %>% dplyr::filter(Dataset == 'Test', Clock_Status == 'Good'), method = 'loess') + facet_wrap(~ Gene_Name) + xlim(xlim_l,xlim_u)

  return(time_vs_exp_plot)
}

#' Assess precision of time predictions
#'
#' Useful when assessing the structure of the test data and the precision of \code{circadianTT} phase estimates
#'
#' @param object list containing circadianTT training and test models following \code{train_model} and \code{test_model} respectively
#' @param ymin lower limit for y axis. Helps with visualisation when samples cluster in a particular part of the day (eg biopsy samples taken mostly during day time)
#' @param ymax upper limit for y axis
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns test data projection onto the first 4 PCs vs estimated time
#' @export
#'

plotPCs_test <- function(object, ymin = 0, ymax = 24) {

  test_exp <- object[['Test_Data']][['Test_Exp_Data']]
  test_exp_intergene <- apply(test_exp, 2, function(x) (x-mean(x))/sd(x))
  test_exp_intergene_centered <- test_exp_intergene - apply(test_exp_intergene, 1, mean)

  test_exp_svd <- svd(test_exp_intergene_centered)
  test_projections <- t(test_exp_svd$u[,1:4]) %*% test_exp_intergene_centered

  opar <- par(no.readonly = TRUE)
  par(mfrow = c(2,2))
  plot(test_projections[1,], object[['Test_Data']][['Results_df']]$time_1st_peak, ylim = c(ymin, ymax), xlab = 'PC1', ylab = 'Corrected Time')
  plot(test_projections[2,], object[['Test_Data']][['Results_df']]$time_1st_peak, ylim = c(ymin, ymax), xlab = 'PC2', ylab = 'Corrected Time')
  plot(test_projections[3,], object[['Test_Data']][['Results_df']]$time_1st_peak, ylim = c(ymin, ymax), xlab = 'PC3', ylab = 'Corrected Time')
  plot(test_projections[4,], object[['Test_Data']][['Results_df']]$time_1st_peak, ylim = c(ymin, ymax), xlab = 'PC4', ylab = 'Corrected Time')

  par(opar)
}

#' Cross-validation results on the training data
#'
#' Visualisation of results following \code{train_cv} function. Plots prediction error by groups
#'
#' @param cv_object list containing the output of \code{train_cv} function
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns prediction error and theta estimate for the CV groups
#' @export
#'

plot_cv_res <- function(cv_object) {
  opar <- par(no.readonly = TRUE)
  par(mfrow = c(1,2))
  pred_errors <- purrr::map(cv_object, as_mapper(~ .x$Test_Data$Results_df$Pred_Error))
  pred_errors_plot_df <- data.frame(PredError = unlist(pred_errors, use.names = FALSE),
                                    Condition = rep(names(pred_errors), times = purrr::map_dbl(pred_errors, length)))
  a_ordered <- with(pred_errors_plot_df, reorder(Condition, PredError, median))
  boxplot(PredError ~ a_ordered, xaxt = 'n', xlab = 'Condition', ylab = 'PredError', main = 'Prediction Error / Cross-Validated', data = pred_errors_plot_df)
  axis(side = 1, labels = FALSE)
  text(x = 1:length(levels(a_ordered)), y = par("usr")[3] - 0.75, labels = levels(a_ordered),
       xpd = NA, srt = 35, cex = 0.7)

  thetas <- purrr::map(cv_object, as_mapper(~ .x$Test_Data$Thetas_Test))
  thetas_plot_df <- data.frame(Theta = unlist(thetas, use.names = FALSE),
                               Condition = rep(names(thetas), times = purrr::map_dbl(thetas, length)))
  a_ordered <- with(thetas_plot_df, reorder(Condition, Theta, median))
  boxplot(Theta ~ a_ordered, xaxt = 'n', xlab = 'Condition', ylab = 'Theta', main = 'Theta / Cross-Validated', data = thetas_plot_df)
  axis(side = 1, labels = FALSE)
  text(x = 1:length(levels(a_ordered)), y = par("usr")[3] - 0.05, labels = levels(a_ordered),
       xpd = NA, srt = 35, cex = 0.7)
  par(opar)
}

#' Cross-validation results on the training data
#'
#' Visualisation of results following \code{train_cv} function. Plots prediction error against actual time.
#' This function can be useful when systematic deviations (eg chronotype) are suspected, in which case error will be adjusted by the median.
#' Care should be taken otherwise, as this could also mask the algorithm's systematic error, therefore \code{plot_deviation_cv_original} and \code{plot_cv_res} are
#' the recommended first steps
#'
#' @param list_cv list containing the output of \code{train_cv} function
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns \code{circadianTT} prediction error plotted against the actual sample time
#' @export
#'

plot_deviation_cv_corrected <- function(list_cv) {
  pred_errors <- purrr::map(list_cv, as_mapper(~ .x$Test_Data$Results_df$Pred_Error - median(.x$Test_Data$Results_df$Pred_Error)))
  actual_times <- purrr::map(list_cv, as_mapper(~ .x$Test_Data$Results_df$Actual_Time))
  names <- factor(rep(names(pred_errors), purrr::map(pred_errors, length)), levels = unique(names(pred_errors)))
  errors <- unlist(pred_errors)
  times <- unlist(actual_times)

  opar <- par(no.readonly = TRUE)
  par(mar=c(5, 4, 4, 8), xpd=TRUE)
  plot(times, errors, pch = 19, col = names, xlab = 'Actual Time', ylab = 'Error', main = 'Error corrected for deviation')
  legend(x = "topright", inset = c(-0.2,0), box.lwd = 2 , title="Group", cex = 0.75,
         legend=levels(names), pch=16, col=unique(names))
  par(opar)
}

#' Cross-validation results on the training data
#'
#' Visualisation of results following \code{train_cv} function. Plots prediction error against actual time.
#'
#' @param list_cv list containing the output of \code{train_cv} function
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns \code{circadianTT} prediction error plotted against the actual sample time
#' @export
#'

plot_deviation_cv_original <- function(list_cv) {
  pred_errors <- purrr::map(list_cv, as_mapper(~ .x$Test_Data$Results_df$Pred_Error))
  actual_times <- purrr::map(list_cv, as_mapper(~ .x$Test_Data$Results_df$Actual_Time))
  names <- factor(rep(names(pred_errors), purrr::map(pred_errors, length)), levels = unique(names(pred_errors)))
  errors <- unlist(pred_errors)
  times <- unlist(actual_times)

  opar <- par(no.readonly = TRUE)
  par(mar=c(5, 4, 4, 8), xpd=TRUE)
  plot(times, errors, pch = 19, col = names, xlab = 'Actual Time', ylab = 'Error', main = 'Original Cross Validation Error')
  legend(x = "topright", inset = c(-0.2,0), box.lwd = 2 , title="Group", cex = 0.75,
         legend=levels(names), pch=16, col=unique(names))
  par(opar)
}

#' Display the results of cosinor rhythmicity analysis
#'
#' Visualisation of results following \code{choose_genes_tt} function
#'
#' @param object list containing the output of \code{train_model} and \code{choose_genes_tt} functions
#' @param probes_of_interest genes / features of interest to highlight
#' @param info_used plot style to output. Must be one of the \code{'ranks'} or \code{'original'}. Choosing \code{'ranks'} allows for more interactivity using \code{ggplotly}
#' @param organism this is added for convenience. Will attempt to use \code{gprofiler2::gconvert} to get Entrez gene symbols for more user-friendly visualisation. Therefore, the argument format has to be consistent with that.
#'
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns \code{plot_ly} or \code{ggplot} object depending on the \code{info_used} selection
#' @export
#'

display_rhythmicity_results <- function(object, probes_of_interest, info_used = 'ranks', organism) {
  if(missing(probes_of_interest)) probes_of_interest <- c()
  check_pkg('gprofiler2', bioc = FALSE)
  if (info_used == 'ranks') {

    if(is.null(suppressMessages(gprofiler2::gconvert(rownames(object$Rhythmicity_Results), organism = organism, target = 'ENTREZGENE', mthreshold = 1, filter_na = FALSE)$target))) {Names <- rownames(object$Rhythmicity_Results)}
    else {Names <- gprofiler2::gconvert(rownames(object$Rhythmicity_Results), organism = organism, target = 'ENTREZGENE', mthreshold = 1, filter_na = FALSE)$target}
    object$Rhythmicity_Results$Gene_Name <- Names

    aa <- object$Rhythmicity_Results
    a1 <- ggplot(dplyr::filter(aa, rank_pval < 2500 & rank_rsquared < 2500), aes(x = rank_pval, y = rank_rsquared, text = paste("Gene:", Gene, '<br>ENTREZ:', Gene_Name))) +
      geom_point(aes(colour = rank_sum), alpha = (1/2)) +
      scale_colour_gradient(high = "white", low = "black") +
      geom_point(data = dplyr::filter(aa, Gene %in% probes_of_interest),
                 aes(x = rank_pval, y = rank_rsquared),
                 color='red',
                 size=2) +
      geom_text(data = dplyr::filter(aa, Gene %in% probes_of_interest),
                aes(x = rank_pval, y = rank_rsquared, label = Gene), check_overlap = TRUE)
    ggplotly(a1)
  }
  else if (info_used == 'original') {

    if(is.null(suppressMessages(gprofiler2::gconvert(rownames(object$Rhythmicity_Results), organism = organism, target = 'ENTREZGENE', mthreshold = 1, filter_na = FALSE)$target))) {Names <- rownames(object$Rhythmicity_Results)}
    else {Names <- gprofiler2::gconvert(rownames(object$Rhythmicity_Results), organism = organism, target = 'ENTREZGENE', mthreshold = 1, filter_na = FALSE)$target}
    object$Rhythmicity_Results$Gene_Name <- Names

    aa <- object$Rhythmicity_Results
    entrez_names <- aa %>% dplyr::filter(Gene %in% probes_of_interest) %>% pull(Gene_Name)

    a1 <- ggplot(aa, aes(x = Pval, y = Rsquared, text = paste("Gene:", Gene)))  +
      geom_point(aes(colour = rank_sum), alpha = (1/2)) +
      scale_x_continuous(trans = "log10") +
      scale_colour_gradient2(high = "red", low = "green4", mid = 'yellow', midpoint = quantile(aa$rank_sum, 0.35)) +
      geom_point(data = dplyr::filter(aa, Gene %in% probes_of_interest),
                 aes(x = Pval, y = Rsquared),
                 color='black',
                 size=2) +
      ggrepel::geom_text_repel(data = dplyr::filter(aa, Gene %in% probes_of_interest),
                               aes(x = Pval, y = Rsquared, label = entrez_names))


    return(a1)
  }
}

#' Log Threshold selection
#'
#' This will iterate though multiple values of logthresh and save the relevant metrics for each. Can be used to aid in selecting an appropriate threshold. See \code{?choose_logthresh_plot} for further visualisation
#'
#' @param object list containing the output of \code{train_model} and \code{test_model} functions
#' @param max_log max value of logthresh to be considered. Generally recommended to check \code{[0,-12]} range
#' @param min_log min value of logthresh to be considered
#' @param by_step step size for logthresh optimisation. We have found -0.5 to be sufficient, however users may benefit from more granular data depending on the application
#' @param train_or_test whether the simulation should be run on \code{'train'} or \code{'test'} data
#'
#'
#' @author Vadim Vasilyev
#'
#'
#' @return returns \code{data.frame} containing the results of log threshold selection
#' @export
#'

choose_logthresh <- function(object, max_log = 0, min_log = -12, by_step = -1, train_or_test = 'test') {

  xx <- data.frame(matrix(NA, nrow = length(seq(max_log,min_log,by=by_step)), ncol = 6))
  colnames(xx) <- c('Perc_Flat', 'PeakNum_Ratio', 'MaxLik_Ratio', 'Mean_Theta', 'Median_Theta', 'FlatContrib')
  iter <- 1

  for (i in seq(max_log,min_log,by=by_step)) {
    logthresh <- i
    cat('Calculating for LogThresh ', logthresh, '\n')

    if (train_or_test == 'test') {
      a_int <- get_final_likelis_test(object, logthresh)
      a_int <- theta_calc_test(a_int)
      a_int <- calc_flat_theta_contrib_test(a_int)
      a_int <- second_peaks_fun_test(a_int)
      data_df <- a_int[['Test_Data']][['Results_df']]

      results_vec <- c()
      perc_flat <- data_df$PercFlat
      perc_flat_constraint <- sum(perc_flat == 100) / dim(object[['Test_Data']][['Averaged_Likelis_Post_Thresh_Test']])[2]
      peaknum_ratio <- round(table(data_df$npeaks)['2']/dim(object[['Test_Data']][['Averaged_Likelis_Post_Thresh_Test']])[2],2)
      maxlik_ratio <- log(mean(exp(data_df$max_1st_peak)/exp(data_df$max_2nd_peak), na.rm = TRUE))
      mean_theta <- round(mean(data_df$Theta),3)
      median_theta <- round(median(data_df$Theta),3)
      mean_flat_theta_contrib <- round(mean(data_df$FlatContrib),2)

      results_vec[1] <- perc_flat_constraint
      results_vec[2] <- peaknum_ratio
      results_vec[3] <- maxlik_ratio
      results_vec[4] <- mean_theta
      results_vec[5] <- median_theta
      results_vec[6] <- mean_flat_theta_contrib / 100

    } else if (train_or_test == 'train') {
      a_int <- get_final_likelis_train(object, logthresh)
      a_int <- theta_calc_train(a_int)
      a_int <- calc_flat_theta_contrib_train(a_int)
      a_int <- second_peaks_fun_train(a_int)
      data_df <- a_int[['Train_Data']][['Results_df']]

      results_vec <- c()
      perc_flat <- data_df$PercFlat
      perc_flat_constraint <- sum(perc_flat == 100) / dim(object[['Train_Data']][['Averaged_Likelis_Post_Thresh_Train']])[2]
      peaknum_ratio <- round(table(data_df$npeaks)['2']/dim(object[['Train_Data']][['Averaged_Likelis_Post_Thresh_Train']])[2],2)
      maxlik_ratio <- log(mean(exp(data_df$max_1st_peak)/exp(data_df$max_2nd_peak), na.rm = TRUE))
      mean_theta <- round(mean(data_df$Theta),3)
      median_theta <- round(median(data_df$Theta),3)
      mean_flat_theta_contrib <- round(mean(data_df$FlatContrib),2)

      results_vec[1] <- perc_flat_constraint
      results_vec[2] <- peaknum_ratio
      results_vec[3] <- maxlik_ratio
      results_vec[4] <- mean_theta
      results_vec[5] <- median_theta
      results_vec[6] <- mean_flat_theta_contrib / 100
    }

    xx[iter,] <- results_vec
    iter <- iter + 1

  }

  xx$LogThresh <- seq(max_log,min_log,by=by_step)
  return(xx)

}

#' Log Threshold selection plot
#'
#' Plots the results of \code{choose_logthresh} function
#'
#' @param choose_logthresh_df \code{data.frame} containing the output of \code{choose_logthresh} function
#' @param cap parameter used for clarity of visualisation. Ratio of LogLik for 1st and 2nd peaks
#' @param perc_flat option to target a particular proportion of samples with flat likelihoods (ie their peak is below the currently selected LogThresh)
#'
#' @author Vadim Vasilyev
#'
#'
#' @return returns \code{ggplot} object
#' @export
#'

choose_logthresh_plot <- function(choose_logthresh_df, cap = 10, perc_flat = 0.02) {
  results_df <- choose_logthresh_df

  suggested_logthresh <- results_df %>% dplyr::mutate(Is_True = Perc_Flat < perc_flat) %>% dplyr::group_by(Is_True) %>%
    dplyr::mutate(First_True = row_number()) %>% dplyr::ungroup() %>% dplyr::filter(Is_True == TRUE & First_True == 1) %>% pull(LogThresh)

  p1 <- results_df %>% dplyr::mutate(MaxLik_Ratio_capped = ifelse(MaxLik_Ratio > cap,cap,MaxLik_Ratio)) %>% dplyr::mutate(MaxLik_Ratio_trans = MaxLik_Ratio_capped/cap)  %>%
    ggplot(aes(x = LogThresh, y = Perc_Flat, group = 1))+
    geom_line(aes(color = "Sample % with Flat LogLik"), linewidth = 1.05) +
    geom_line(aes(y = PeakNum_Ratio, color = "Sample % with 2 Peaks"), linewidth = 1.05) +
    geom_line(aes(y = Mean_Theta, color = "Mean Theta for all Samples"), linewidth = 1.05) +
    geom_line(aes(y = FlatContrib, color = "Flat Region Contribution to Theta"), linewidth = 1.05) +
    geom_line(aes(y = MaxLik_Ratio_trans, color = "Mean MaxLik Ratio for Samples with 2 Peaks"), linewidth = 1.5) +
    scale_x_reverse() + scale_y_continuous(sec.axis = sec_axis(~.*cap, name = "MaxLik Ratio")) +
    labs(x = "LogThresh", y = "Value", color = "") +
    scale_color_manual(values = c("orange2", "gray30", "purple", "green", "red")) + theme(legend.position = c(0.25, 0.90), legend.background=element_blank())
  # + geom_vline(xintercept = suggested_logthresh, col = 'blue')
  # + geom_label(
  #   label=paste0('LogThresh Suggested is: ', suggested_logthresh),
  #   x=10,
  #   y=0.2,
  #   label.padding = unit(0.55, "lines"), # Rectangle size around label
  #   label.size = 0.35,
  #   color = "black",
  #   fill="#69b3a2"
  # )
  return(p1)
}

#' Likelihoods of individual samples
#'
#' This displays likelihoods (for all the local projections) for individual samples. Can be useful for diagnostics and troubleshooting the model
#'
#' @param object list containing circadianTT training and test models following \code{train_model} and \code{test_model} respectively
#' @param sample_num number of sample to be displayed
#' @param logthresh which log threshold should be used for visualisation
#' @param train_or_test is the sample coming from \code{train} or \code{test} data
#'
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns a plot of likelihood curves produced using \code{graphics::matplot}
#' @export
#'

plot_raw_likelis <- function(object, sample_num, logthresh, train_or_test = 'test') {
  if (train_or_test == 'test') {likelis_array <- object[['Test_Data']][['Test_Likelihood_Array']]}
  else if (train_or_test == 'train') {likelis_array <- object[['Train_Data']][['Train_Likelihood_Array']]}

  likeli_mat <- likelis_array[ ,sample_num, ]
  likeli_mat_adjusted <- pmax(likeli_mat, logthresh)

  opar <- par(no.readonly = TRUE)
  matplot(1:dim(likeli_mat_adjusted)[1]/dim(likeli_mat_adjusted)[1]*24, likeli_mat_adjusted,
          type = 'l', xlab = 'Time', ylab = 'Raw Truncated Likelihood',
          main = paste('Sample_',sample_num))
  averaged_likeli <- apply(likeli_mat_adjusted, 1, mean)
  matlines(1:dim(likeli_mat_adjusted)[1]/dim(likeli_mat_adjusted)[1]*24, averaged_likeli, type = "l", lty = 1, col = 'black', lwd = 3)
  par(opar)
}

#' Theta calculation of individual samples
#'
#' This displays \code{theta} and flat likelihood percentage calculation for individual samples. Can be useful for diagnostics and troubleshooting the model
#'
#' @param object list containing circadianTT training and test models following \code{train_model} and \code{test_model} respectively
#' @param sample_num number of sample to be displayed
#' @param logthresh which log threshold should be used for visualisation
#' @param train_or_test is the sample coming from \code{train} or \code{test} data
#'
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns plots of theta calculation curves produced using standard \code{graphics::plot}
#' @export
#'

plot_ind_curve <- function(object, sample_num, logthresh, train_or_test = 'test') {
  epsilon <- object[['Train_Data']][['epsilon']]
  eta <- object[['Train_Data']][['eta']]
  if (train_or_test == 'test') {likelis_array <- object[['Test_Data']][['Test_Likelihood_Array']]}
  else if (train_or_test == 'train') {likelis_array <- object[['Train_Data']][['Train_Likelihood_Array']]}

  num_time_points <- dim(likelis_array)[3]
  for (i in 1:num_time_points) {
    likelis_array[,,i] <- pmax(likelis_array[,,i], logthresh)
  }

  averaged_likelis_rescaled <- t(apply(likelis_array, c(1,2), mean))
  num_samples <- dim(averaged_likelis_rescaled)[1]
  num_points <- dim(averaged_likelis_rescaled)[2]
  ind_ts <- exp(averaged_likelis_rescaled[sample_num,])
  ind_ts <- shift_ts(ind_ts, round(num_points/2))
  ind_lrf_curve <- ind_ts / max(ind_ts)

  curr_curve <- suppressWarnings(eta*(1 + epsilon + cos(2*pi*((1:num_points)/num_points - which(ind_lrf_curve == max(ind_lrf_curve))/num_points))))
  lrf_curve_spline <- predict(periodicSpline(1:num_points,ind_lrf_curve, period = num_points),seq(1,num_points,length.out = 1000))
  curve_spline <- predict(periodicSpline(1:num_points,curr_curve, period = num_points),seq(1,num_points,length.out = 1000))

  theta <- sum(lrf_curve_spline$y > curve_spline$y) / length(lrf_curve_spline$y)

  theta_index <- which(lrf_curve_spline$y > curve_spline$y)
  flat_regions <- which(abs(diff(lrf_curve_spline$y)) < 1e-4)
  flat_contribution <- sum(theta_index %in% flat_regions) / length(theta_index) * 100
  flat_regions_in_theta <- flat_regions[flat_regions %in% theta_index]
  nonflat_regions_in_theta <- theta_index[!theta_index %in% flat_regions]
  # Plotting everything
  opar <- par(no.readonly = TRUE)
  par(mfrow = c(2,1))
  plot(lrf_curve_spline, type = 'l', lwd = 2, col = 'blue', main = paste0('Plots for sample ',sample_num), xlab = '', ylab = 'Likelihood', ylim = c(0,1))
  lines(curve_spline, type = 'l', lwd = 2, col = 'black')
  mtext(paste("Theta:", round(theta,3)), side=3)

  plot(lrf_curve_spline, type = 'l', lwd = 2, col = 'blue', main = 'Finding Contribution of Flat region to Theta', xlab = '', ylab = 'Likelihood', ylim = c(0,1))
  points(lrf_curve_spline$x[flat_regions_in_theta], lrf_curve_spline$y[flat_regions_in_theta], col = 'red')
  points(lrf_curve_spline$x[nonflat_regions_in_theta], lrf_curve_spline$y[nonflat_regions_in_theta], col = 'green')
  mtext(paste('Flat region Contribution to Theta is:', round(flat_contribution,1)), side = 3)

  res = recordPlot()
  par(opar)

  return(res)

}

#' Rhythmicity analysis of training data
#'
#' Calculates rhythmicity results for the training data
#'
#' @param object list containing circadianTT training and test models following \code{train_model} and \code{test_model} respectively
#' @param grouping_var how samples should be grouped for rhythmicity analysis. Should be either provided manually or be one of the following: \code{'Group_1'}, \code{'Group_2'}, \code{'Group_3'} or \code{'Replicate'}
#' @param method method used for rhythmicity analysis. Default is \code{'population'} as in \url{https://tbiomed.biomedcentral.com/articles/10.1186/1742-4682-11-16}
#' @param parallel if TRUE, parallel computation using \code{foreach} and \code{doParallel} packages will be used. Default is TRUE
#' @param cores if using parallel computation, how many cores should be used. Default is 4
#'
#' @author Vadim Vasilyev
#'
#' @references
#'
#' Cornelissen, G., 2014. Cosinor-based rhythmometry. Theoretical Biology and Medical Modelling, 11(1), pp.1-24.
#'
#' @return Returns plots of theta calculation curves produced using standard \code{graphics::plot}
#' @export
#'

choose_genes_tt <- function(object, grouping_var = 'Group_1', method = 'population', parallel = TRUE, cores = 4){
  message(paste0('Selection will be done using ', ifelse(length(grouping_var) > 1, 'Manual',grouping_var)))
  data <- object[['Full_Original_Data']]
  time_vec <- object[['Metadata']][['Train']][['Time']]
  entrained_results_df <- data.frame(Pval = rep(NA,length(rownames(data))), Phase = rep(NA,length(rownames(data))),
                                     rAMP = rep(NA,length(rownames(data))), Rsquared = rep(NA,length(rownames(data))))
  rownames(entrained_results_df) <- rownames(data)

  if(length(grouping_var) > 1) {
    group_vec <- grouping_var
  } else if(grouping_var == 'Group_1') {
    group_vec <- object[['Metadata']][['Train']][['Group_1']]
  } else if(grouping_var == 'Group_2') {
    group_vec <- object[['Metadata']][['Train']][['Group_2']]
  } else if(grouping_var == 'Group_3') {
    group_vec <- object[['Metadata']][['Train']][['Group_3']]
  } else if(grouping_var == 'Replicate') {
    group_vec <- object[['Metadata']][['Train']][['Replicate']]
  } else {
    stop("Grouping variable specified not found in Metadata")
  }

  if (parallel == TRUE) {
    my.cluster <- parallel::makeCluster(
      cores,
      type = "PSOCK"
    )
    doParallel::registerDoParallel(cl = my.cluster)

    results_df <- foreach(i = 1:length(rownames(data)), .combine = 'rbind', .inorder = TRUE, .packages = c('cosinor2','psych','tidyverse'), .export = c('data','group_vec','time_vec')) %dopar% {
      curr_gene <- rownames(data)[i]
      expression_df <- data.frame(Expression = data[curr_gene,], Time = time_vec, Group = factor(group_vec), Replicate = factor(object[['Metadata']][['Train']][['Replicate']])) %>%
        tidyr::unite('Group_rep', c(Group,Replicate), remove = TRUE, sep = '|')
      expression_df <- expression_df %>% tidyr::pivot_wider(names_from = Time, values_from = Expression) %>%
        tibble::column_to_rownames('Group_rep')
      times <- as.numeric(colnames(expression_df))

      pop_cosinor <- population.cosinor.lm(expression_df, times, period = 24, plot = FALSE)
      population_cos_rAMP <- pop_cosinor$coefficients$Amplitude / pop_cosinor$coefficients$MESOR
      population_cos_Phase <- circadian.mean(unlist(lapply(pop_cosinor$single.cos, function(x) round(-(correct.acrophase(x)) / (2*pi/24),2))))
      population_cos_MESOR <- pop_cosinor$coefficients$MESOR
      population_cos_Phase_lower <- round(pop_cosinor$conf.ints[2,3] / (2*pi/24),2)
      population_rhythm_pval <- cosinor.detect(pop_cosinor)[4]
      population_r_squared <- cosinor.PR(pop_cosinor)$`Percent rhythm`

      data.frame(Pval = population_rhythm_pval, Phase = population_cos_Phase,
                 MESOR = population_cos_MESOR, rAMP = population_cos_rAMP, Rsquared = population_r_squared)

    }

    rownames(results_df) <- rownames(data)
    results_df$Pval.adj <- round(stats::p.adjust(results_df$Pval, "BH"), 3)
    results_df$Gene <- rownames(results_df)
    results_df <- results_df %>% dplyr::mutate(rank_pval = base::rank(Pval, ties.method = 'random'),
                                               rank_rsquared = base::rank(-Rsquared, ties.method = 'random')) %>%
      dplyr::mutate(rank_sum = rank_pval + rank_rsquared)
    object[['Rhythmicity_Results']] <- results_df
    return(object)

  } else {
    pb <- txtProgressBar(min = 0, max = length(rownames(data)), style = 3, width = 50, char = "=")
    for (i in 1:length(rownames(data))) {
      curr_gene <- rownames(data)[i]
      expression_df <- data.frame(Expression = data[curr_gene,], Time = time_vec, Group = factor(group_vec))
      expression_df <- expression_df %>% tidyr::pivot_wider(names_from = Time, values_from = Expression) %>%
        tibble::column_to_rownames('Group')
      times <- as.numeric(colnames(expression_df))
      pop_cosinor <- quiet(population.cosinor.lm(expression_df, times, period = 24, plot = FALSE))
      population_cos_rAMP <- pop_cosinor$coefficients$Amplitude / pop_cosinor$coefficients$MESOR
      population_cos_Phase <- psych::circadian.mean(unlist(lapply(pop_cosinor$single.cos, function(x) round(-(correct.acrophase(x)) / (2*pi/24),2))))
      population_rhythm_pval <- cosinor.detect(pop_cosinor)[4]
      population_r_squared <- cosinor.PR(pop_cosinor)$`Percent rhythm`
      entrained_results_df[curr_gene ,'Pval'] <- population_rhythm_pval
      entrained_results_df[curr_gene ,'Phase'] <- population_cos_Phase
      entrained_results_df[curr_gene ,'rAMP'] <- population_cos_rAMP
      entrained_results_df[curr_gene ,'Rsquared'] <- population_r_squared

      Sys.sleep(0.05)
      setTxtProgressBar(pb, i)
    }

    close(pb)
    entrained_results_df$Pval.adj <- round(stats::p.adjust(entrained_results_df$Pval, "BH"), 3)
    entrained_results_df$Gene <- rownames(entrained_results_df)
    entrained_results_df <- entrained_results_df %>% dplyr::mutate(rank_pval = rank(Pval, ties.method = 'random'),
                                                                   rank_rsquared = rank(desc(Rsquared), ties.method = 'random')) %>%
      dplyr::mutate(rank_sum = rank_pval + rank_rsquared)
    object[['Rhythmicity_Results']] <- entrained_results_df
    return(object)
  }
  parallel::stopCluster(cl = my.cluster)
}

#' Covariance matrix ellipsoid volume
#'
#' This displays the volume (area) of the hyper-ellipse at a given confidence level (\url{https://datavis.ca/papers/ellipses.pdf}). Can be useful for diagnostics and further analysis of the covariance matrix interpolation
#'
#' @param object list containing circadianTT training and test models following \code{train_model}
#' @param alpha confidence level used for volume calculation
#'
#' @author Vadim Vasilyev
#'
#'
#' @return Returns the plot of volume of hyper-ellipse for all the local projections using \code{graphics::matplot}
#' @export
#'

check_cov_interp <- function(object, alpha = 0.95) {
  dims <- object$PC_Num
  cov_mat_list <- object$Projections$Fitted_MVN_Interpolated[(dims+1):(dims+dims^2), , ]
  res_mat <- matrix(NA, nrow = dim(cov_mat_list)[3], ncol = dim(cov_mat_list)[2])
  rownames(res_mat) <- unlist(dimnames(object$Projections$Fitted_MVN_Interpolated)[3])
  times_x <- dimnames(cov_mat_list)[[2]]
  colnames(res_mat) <- times_x

  for (i in 1:dim(cov_mat_list)[3]) {
    for (j in 1:dim(cov_mat_list)[2]) {
      res_mat[i,j] <- suppressWarnings(volume_func(matrix(cov_mat_list[ ,j,i], nrow = dims), alpha = alpha, dims = dims))
    }
  }
  opar <- par(no.readonly = TRUE)
  matplot(t(res_mat), type = 'l', lwd = 2, lty=1:5, col=1:6, xlab = 'Index', ylab = 'Volume', main = 'Covariance matrix interpolation')
  legend('topright', lty=1:5, col=1:6, rownames(res_mat), lwd = 2, cex = 0.5)

  object[['Projections']][['Fitted_MVN_Interpolated_Volume']] <- res_mat
  par(opar)
  return(object)
}
