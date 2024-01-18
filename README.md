
<!-- README.md is generated from README.Rmd. Please edit that file -->

# TimeTeller

<!-- badges: start -->
<!-- badges: end -->

The goal of TimeTeller is to …

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

What is special about using `README.Rmd` instead of just `README.md`?
You can include R chunks like so:

``` r
summary(cars)
#>      speed           dist       
#>  Min.   : 4.0   Min.   :  2.00  
#>  1st Qu.:12.0   1st Qu.: 26.00  
#>  Median :15.0   Median : 36.00  
#>  Mean   :15.4   Mean   : 42.98  
#>  3rd Qu.:19.0   3rd Qu.: 56.00  
#>  Max.   :25.0   Max.   :120.00
```

You’ll still need to render `README.Rmd` regularly, to keep `README.md`
up-to-date. `devtools::build_readme()` is handy for this.

You can also embed plots, for example:

<img src="man/figures/README-pressure-1.png" width="100%" />

In that case, don’t forget to commit and push the resulting figure
files, so they display on GitHub and CRAN.
