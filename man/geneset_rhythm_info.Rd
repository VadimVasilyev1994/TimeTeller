% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Plotting_and_Diagnostics.R
\name{geneset_rhythm_info}
\alias{geneset_rhythm_info}
\title{Rhythmicity analysis of selected geneset}
\usage{
geneset_rhythm_info(
  object,
  geneset,
  labels,
  group1,
  group2,
  group3,
  replicate,
  method = "population"
)
}
\arguments{
\item{object}{list containing TimeTeller rhythmicity results following \code{choose_genes_tt}}

\item{geneset}{geneset of interest}

\item{labels}{genes to highlight on the resulting polar plot}

\item{group1}{if only a subset of Group_1 (Metadata provided) should be used}

\item{group2}{if only a subset of Group_2 (Metadata provided) should be used}

\item{group3}{if only a subset of Group_3 (Metadata provided) should be used}

\item{replicate}{if only a subset of Replicate (Metadata provided) should be used}

\item{method}{method used for rhythmicity analysis. Default is \code{'population'} as in \url{https://tbiomed.biomedcentral.com/articles/10.1186/1742-4682-11-16}}
}
\value{
Returns an array containing single cosinor results (MESOR, Amp and Rhythmicity test Pval) and the polar plot with summary info
}
\description{
Convenience function for single cosinor and visualisation of the selected geneset
}
\references{
Cornelissen, G., 2014. Cosinor-based rhythmometry. Theoretical Biology and Medical Modelling, 11(1), pp.1-24.
}
\author{
Vadim Vasilyev
}
