#' Panda data
#'
#' Mice were subjected to ALF or TRF feeding schedule and gene expression changes were examined in samples taken from 22 organs
#' and brain regions collected every 2 hours over a 24-hour period (\url{https://pubmed.ncbi.nlm.nih.gov/36599299/}). Here we
#' will look at the subset of that extensive study, in particular TRF and non-brain regions. Expression profiles reported underwent
#' basic library depth filtering and were CPM normalised. This left 378 samples and for memory and speed considerations
#' 1000 randomly selected genes together with clock genes were selected for further analysis.
#'
#' @format A list containing the following:
#' \describe{
#'  \item{expr_mat}{CPM normalised expression matrix}
#'  \item{group}{Organ (order corresponds to columns in the expression matrix)}
#'  \item{time}{ZT sampling time (order corresponds to columns in the expression matrix)}
#'  \item{replicate}{'1' or '2' (order corresponds to columns in the expression matrix)}
#'  \item{genes_used}{Clock gene probes used for training the model}
#' }
#' @source Deota et. al (2023)
#'
#' @references
#'
#' Deota, S., Lin, T., Chaix, A., Williams, A., Le, H., Calligaro, H., Ramasamy, R., Huang, L. and Panda, S., 2023. Diurnal transcriptome landscape of a multi-tissue response to time-restricted feeding in mammals. Cell metabolism, 35(1), pp.150-165.
#'
"panda_data"
