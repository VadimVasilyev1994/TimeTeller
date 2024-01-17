
<!-- README.md is generated from README.Rmd. Please edit that file -->

# TimeTeller

<!-- badges: start -->
<!-- badges: end -->

TimeTeller is a supervised machine learning tool that analyses the local
circadian clock as a system. It aims to estimate the circadian clock
phase and the level of dysfunction from a single sample by modelling the
multi-dimensional state of the clock.

*TimeTeller* package implements the algorithm including methodology
improvements and tools for visualisation.

## Installation

You can install the development version of TimeTeller from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("VadimVasilyev1994/TimeTeller")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(devtools)
#> Loading required package: usethis
suppressWarnings(devtools::install_github("VadimVasilyev1994/TimeTeller", force = TRUE, quiet = TRUE))
#> Installing 13 packages: rlang, glue, cli, vctrs, fansi, stringi, yaml, Rcpp, dplyr, checkmate, data.table, matrixStats, robustbase
#> package 'rlang' successfully unpacked and MD5 sums checked
#> package 'glue' successfully unpacked and MD5 sums checked
#> package 'cli' successfully unpacked and MD5 sums checked
#> package 'vctrs' successfully unpacked and MD5 sums checked
#> package 'fansi' successfully unpacked and MD5 sums checked
#> package 'stringi' successfully unpacked and MD5 sums checked
#> package 'yaml' successfully unpacked and MD5 sums checked
#> package 'Rcpp' successfully unpacked and MD5 sums checked
#> package 'dplyr' successfully unpacked and MD5 sums checked
#> package 'checkmate' successfully unpacked and MD5 sums checked
#> package 'data.table' successfully unpacked and MD5 sums checked
#> package 'matrixStats' successfully unpacked and MD5 sums checked
#> package 'robustbase' successfully unpacked and MD5 sums checked
suppressWarnings(library(TimeTeller))
```
