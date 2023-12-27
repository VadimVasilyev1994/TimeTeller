
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
library(TimeTeller)
#> Warning: replacing previous import 'Matrix::image' by 'graphics::image' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'graphics::layout' by 'plotly::layout' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'ggplot2::last_plot' by 'plotly::last_plot'
#> when loading 'TimeTeller'
#> Warning: replacing previous import 'ggplot2::%+%' by 'psych::%+%' when loading
#> 'TimeTeller'
#> Warning: replacing previous import 'ggplot2::alpha' by 'psych::alpha' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'foreach::when' by 'purrr::when' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'foreach::accumulate' by 'purrr::accumulate'
#> when loading 'TimeTeller'
#> Warning: replacing previous import 'Matrix::cov2cor' by 'stats::cov2cor' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'dplyr::lag' by 'stats::lag' when loading
#> 'TimeTeller'
#> Warning: replacing previous import 'plotly::filter' by 'stats::filter' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'Matrix::toeplitz' by 'stats::toeplitz' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'circular::var' by 'stats::var' when loading
#> 'TimeTeller'
#> Warning: replacing previous import 'circular::sd' by 'stats::sd' when loading
#> 'TimeTeller'
#> Warning: replacing previous import 'Matrix::update' by 'stats::update' when
#> loading 'TimeTeller'
#> Warning: replacing previous import 'Matrix::head' by 'utils::head' when loading
#> 'TimeTeller'
#> Warning: replacing previous import 'Matrix::tail' by 'utils::tail' when loading
#> 'TimeTeller'
#> Warning: replacing previous import 'purrr::transpose' by
#> 'data.table::transpose' when loading 'TimeTeller'
## basic example code
```
