# Gate 1b report

**(b') HOLDS.**

## (b') TCR(0.25)+rw against adaptX, response designs

| cell | adaptX | TCR(0.25)+rw | diff | SE | n | lower 95% |
|---|---|---|---|---|---|---|
| D1-K2cross ε=0.25 | 0.910 | 0.848 | +0.062 | 0.014 | 100 | +0.034 |
| D1-K2cross ε=0.30 | 0.911 | 0.561 | +0.350 | 0.011 | 100 | +0.329 |
| D1-K2cross ε=0.35 | 0.910 | 0.528 | +0.383 | 0.005 | 100 | +0.373 |
| D2-K3par ε=0.25 | 0.865 | 0.744 | +0.121 | 0.013 | 100 | +0.095 |
| D2-K3par ε=0.30 | 0.765 | 0.619 | +0.146 | 0.019 | 100 | +0.109 |
| D2-K3par ε=0.35 | 0.555 | 0.537 | +0.018 | 0.019 | 100 | -0.020 |
| D3-unequal ε=0.25 | 0.888 | 0.882 | +0.006 | 0.011 | 100 | -0.015 |
| D3-unequal ε=0.30 | 0.881 | 0.751 | +0.130 | 0.012 | 100 | +0.106 |
| D3-unequal ε=0.35 | 0.840 | 0.703 | +0.137 | 0.013 | 100 | +0.112 |
| D4-hetero ε=0.25 | 0.911 | 0.877 | +0.034 | 0.011 | 100 | +0.012 |
| D4-hetero ε=0.30 | 0.907 | 0.619 | +0.288 | 0.017 | 100 | +0.255 |
| D4-hetero ε=0.35 | 0.899 | 0.531 | +0.369 | 0.008 | 100 | +0.353 |
| D6-uniform ε=0.25 | 0.828 | 0.907 | -0.079 | 0.016 | 100 | -0.110 |
| D6-uniform ε=0.30 | 0.684 | 0.891 | -0.207 | 0.019 | 100 | -0.244 |
| D6-uniform ε=0.35 | 0.585 | 0.778 | -0.192 | 0.017 | 100 | -0.226 |
| D7-p3 ε=0.25 | 0.910 | 0.806 | +0.104 | 0.017 | 100 | +0.070 |
| D7-p3 ε=0.30 | 0.906 | 0.564 | +0.343 | 0.012 | 100 | +0.319 |
| D7-p3 ε=0.35 | 0.877 | 0.530 | +0.347 | 0.011 | 100 | +0.326 |
| D8-K2par-sep ε=0.25 | 0.999 | 0.937 | +0.061 | 0.016 | 100 | +0.030 |
| D8-K2par-sep ε=0.30 | 0.999 | 0.582 | +0.417 | 0.015 | 100 | +0.388 |
| D8-K2par-sep ε=0.35 | 0.999 | 0.528 | +0.471 | 0.002 | 100 | +0.467 |

## (d) cells where adaptX mean accuracy < 0.85

- D2-K3par ε=0.30: 0.765
- D2-K3par ε=0.35: 0.555
- D3-unequal ε=0.35: 0.840
- D5-leverage ε=0.25: 0.642
- D5-leverage ε=0.30: 0.538
- D5-leverage ε=0.35: 0.527
- D6-uniform ε=0.25: 0.828
- D6-uniform ε=0.30: 0.684
- D6-uniform ε=0.35: 0.585

## (d) cells where a competitor beats adaptX by > 0.03 (diff + 1.96 SE < 0)

- D2-K3par ε=0.30: TA(0.40) ahead by 0.120 (SE 0.017)
- D2-K3par ε=0.35: TA(0.25) ahead by 0.109 (SE 0.019)
- D2-K3par ε=0.35: TA(0.40) ahead by 0.257 (SE 0.019)
- D2-K3par ε=0.35: TCR(0.40) ahead by 0.171 (SE 0.021)
- D2-K3par ε=0.35: TCR(0.40)+rw ahead by 0.148 (SE 0.021)
- D2-K3par ε=0.35: TLE(0.40) ahead by 0.155 (SE 0.026)
- D2-K3par ε=0.35: tcwm(0.40) ahead by 0.063 (SE 0.021)
- D3-unequal ε=0.35: TA(0.40) ahead by 0.035 (SE 0.009)
- D3-unequal ε=0.35: TCR(0.40) ahead by 0.069 (SE 0.011)
- D3-unequal ε=0.35: TCR(0.40)+rw ahead by 0.031 (SE 0.009)
- D3-unequal ε=0.35: TLE(0.40) ahead by 0.068 (SE 0.011)
- D5-leverage ε=0.25: TA(0.40) ahead by 0.141 (SE 0.021)
- D5-leverage ε=0.25: TCR(0.40) ahead by 0.125 (SE 0.021)
- D5-leverage ε=0.25: TCR(0.40)+rw ahead by 0.123 (SE 0.021)
- D5-leverage ε=0.25: tcwm(0.15) ahead by 0.051 (SE 0.020)
- D5-leverage ε=0.25: tcwm(0.25) ahead by 0.095 (SE 0.021)
- D5-leverage ε=0.25: tcwm(0.40) ahead by 0.187 (SE 0.020)
- D5-leverage ε=0.30: TA(0.40) ahead by 0.098 (SE 0.017)
- D5-leverage ε=0.30: TCR(0.40) ahead by 0.084 (SE 0.016)
- D5-leverage ε=0.30: TCR(0.40)+rw ahead by 0.086 (SE 0.017)
- D5-leverage ε=0.30: tcwm(0.05) ahead by 0.089 (SE 0.014)
- D5-leverage ε=0.30: tcwm(0.10) ahead by 0.110 (SE 0.013)
- D5-leverage ε=0.30: tcwm(0.15) ahead by 0.134 (SE 0.014)
- D5-leverage ε=0.30: tcwm(0.25) ahead by 0.161 (SE 0.014)
- D5-leverage ε=0.30: tcwm(0.40) ahead by 0.220 (SE 0.016)
- D5-leverage ε=0.35: tcwm(0.05) ahead by 0.107 (SE 0.013)
- D5-leverage ε=0.35: tcwm(0.10) ahead by 0.133 (SE 0.013)
- D5-leverage ε=0.35: tcwm(0.15) ahead by 0.166 (SE 0.014)
- D5-leverage ε=0.35: tcwm(0.25) ahead by 0.148 (SE 0.013)
- D5-leverage ε=0.35: tcwm(0.40) ahead by 0.177 (SE 0.014)
- D6-uniform ε=0.25: TA(0.25) ahead by 0.078 (SE 0.016)
- D6-uniform ε=0.25: TA(0.40) ahead by 0.079 (SE 0.016)
- D6-uniform ε=0.25: TCR(0.15) ahead by 0.062 (SE 0.014)
- D6-uniform ε=0.25: TCR(0.15)+rw ahead by 0.061 (SE 0.015)
- D6-uniform ε=0.25: TCR(0.25) ahead by 0.079 (SE 0.015)
- D6-uniform ε=0.25: TCR(0.25)+rw ahead by 0.079 (SE 0.016)
- D6-uniform ε=0.25: TCR(0.40) ahead by 0.080 (SE 0.015)
- D6-uniform ε=0.25: TCR(0.40)+rw ahead by 0.079 (SE 0.016)
- D6-uniform ε=0.25: TCRx(0.10) ahead by 0.042 (SE 0.015)
- D6-uniform ε=0.25: TCRx(0.10)+rw ahead by 0.039 (SE 0.015)
- D6-uniform ε=0.25: TCRx(0.25) ahead by 0.079 (SE 0.016)
- D6-uniform ε=0.25: TCRx(0.25)+rw ahead by 0.080 (SE 0.016)
- D6-uniform ε=0.25: TLE(0.05) ahead by 0.050 (SE 0.016)
- D6-uniform ε=0.25: TLE(0.10) ahead by 0.073 (SE 0.015)
- D6-uniform ε=0.25: TLE(0.15) ahead by 0.074 (SE 0.015)
- D6-uniform ε=0.25: TLE(0.25) ahead by 0.079 (SE 0.015)
- D6-uniform ε=0.25: TLE(0.40) ahead by 0.067 (SE 0.015)
- D6-uniform ε=0.25: bisq ahead by 0.067 (SE 0.015)
- D6-uniform ε=0.30: TA(0.25) ahead by 0.205 (SE 0.019)
- D6-uniform ε=0.30: TA(0.40) ahead by 0.225 (SE 0.019)
- D6-uniform ε=0.30: TCR(0.15) ahead by 0.095 (SE 0.019)
- D6-uniform ε=0.30: TCR(0.15)+rw ahead by 0.079 (SE 0.020)
- D6-uniform ε=0.30: TCR(0.25) ahead by 0.210 (SE 0.019)
- D6-uniform ε=0.30: TCR(0.25)+rw ahead by 0.207 (SE 0.019)
- D6-uniform ε=0.30: TCR(0.40) ahead by 0.228 (SE 0.019)
- D6-uniform ε=0.30: TCR(0.40)+rw ahead by 0.225 (SE 0.019)
- D6-uniform ε=0.30: TCRx(0.10) ahead by 0.043 (SE 0.022)
- D6-uniform ε=0.30: TCRx(0.25) ahead by 0.222 (SE 0.019)
- D6-uniform ε=0.30: TCRx(0.25)+rw ahead by 0.220 (SE 0.019)
- D6-uniform ε=0.30: TLE(0.05) ahead by 0.127 (SE 0.019)
- D6-uniform ε=0.30: TLE(0.10) ahead by 0.182 (SE 0.019)
- D6-uniform ε=0.30: TLE(0.15) ahead by 0.210 (SE 0.019)
- D6-uniform ε=0.30: TLE(0.25) ahead by 0.224 (SE 0.019)
- D6-uniform ε=0.30: TLE(0.40) ahead by 0.210 (SE 0.018)
- D6-uniform ε=0.30: bisq ahead by 0.175 (SE 0.022)
- D6-uniform ε=0.30: tcwm(0.15) ahead by 0.045 (SE 0.022)
- D6-uniform ε=0.30: tcwm(0.25) ahead by 0.070 (SE 0.022)
- D6-uniform ε=0.30: tcwm(0.40) ahead by 0.134 (SE 0.023)
- D6-uniform ε=0.35: CTLE ahead by 0.108 (SE 0.026)
- D6-uniform ε=0.35: TA(0.25) ahead by 0.160 (SE 0.016)
- D6-uniform ε=0.35: TA(0.40) ahead by 0.316 (SE 0.014)
- D6-uniform ε=0.35: TCR(0.15) ahead by 0.053 (SE 0.015)
- D6-uniform ε=0.35: TCR(0.25) ahead by 0.208 (SE 0.016)
- D6-uniform ε=0.35: TCR(0.25)+rw ahead by 0.192 (SE 0.017)
- D6-uniform ε=0.35: TCR(0.40) ahead by 0.318 (SE 0.014)
- D6-uniform ε=0.35: TCR(0.40)+rw ahead by 0.313 (SE 0.014)
- D6-uniform ε=0.35: TCRx(0.25) ahead by 0.233 (SE 0.016)
- D6-uniform ε=0.35: TCRx(0.25)+rw ahead by 0.219 (SE 0.017)
- D6-uniform ε=0.35: TLE(0.05) ahead by 0.142 (SE 0.020)
- D6-uniform ε=0.35: TLE(0.10) ahead by 0.218 (SE 0.018)
- D6-uniform ε=0.35: TLE(0.15) ahead by 0.228 (SE 0.017)
- D6-uniform ε=0.35: TLE(0.25) ahead by 0.302 (SE 0.014)
- D6-uniform ε=0.35: TLE(0.40) ahead by 0.298 (SE 0.015)
- D6-uniform ε=0.35: bisq ahead by 0.242 (SE 0.016)
- D6-uniform ε=0.35: tcwm(0.05) ahead by 0.083 (SE 0.019)
- D6-uniform ε=0.35: tcwm(0.10) ahead by 0.064 (SE 0.019)
- D6-uniform ε=0.35: tcwm(0.15) ahead by 0.082 (SE 0.019)
- D6-uniform ε=0.35: tcwm(0.25) ahead by 0.095 (SE 0.017)
- D6-uniform ε=0.35: tcwm(0.40) ahead by 0.133 (SE 0.019)
- D7-p3 ε=0.35: TA(0.40) ahead by 0.032 (SE 0.011)

## Mean accuracy, selected methods

|                        |   adaptX |   adapt |   TCR(0.25)+rw |   TCR(0.40)+rw |   TCR(0.25) |   TCR(0.40) |   TCRx(0.25)+rw |   tcwm(0.40) |   TLE(0.40) |   CTLE |   bisq |
|:-----------------------|---------:|--------:|---------------:|---------------:|------------:|------------:|----------------:|-------------:|------------:|-------:|-------:|
| ('D1-K2cross', 0.25)   |    0.91  |   0.91  |          0.848 |          0.91  |       0.847 |       0.908 |         nan     |        0.845 |       0.909 |  0.902 |  0.879 |
| ('D1-K2cross', 0.3)    |    0.911 |   0.903 |          0.561 |          0.911 |       0.561 |       0.913 |         nan     |        0.853 |       0.906 |  0.774 |  0.761 |
| ('D1-K2cross', 0.35)   |    0.91  |   0.87  |          0.528 |          0.91  |       0.528 |       0.91  |         nan     |        0.859 |       0.912 |  0.65  |  0.727 |
| ('D2-K3par', 0.25)     |    0.865 |   0.798 |          0.744 |          0.816 |       0.734 |       0.78  |         nan     |        0.63  |       0.725 |  0.503 |  0.523 |
| ('D2-K3par', 0.3)      |    0.765 |   0.658 |          0.619 |          0.788 |       0.618 |       0.782 |         nan     |        0.616 |       0.759 |  0.45  |  0.465 |
| ('D2-K3par', 0.35)     |    0.555 |   0.522 |          0.537 |          0.703 |       0.538 |       0.726 |         nan     |        0.618 |       0.71  |  0.403 |  0.418 |
| ('D3-unequal', 0.25)   |    0.888 |   0.906 |          0.882 |          0.908 |       0.886 |       0.906 |         nan     |        0.642 |       0.681 |  0.792 |  0.842 |
| ('D3-unequal', 0.3)    |    0.881 |   0.888 |          0.751 |          0.905 |       0.763 |       0.907 |         nan     |        0.743 |       0.828 |  0.77  |  0.829 |
| ('D3-unequal', 0.35)   |    0.84  |   0.807 |          0.703 |          0.871 |       0.71  |       0.91  |         nan     |        0.865 |       0.909 |  0.651 |  0.819 |
| ('D4-hetero', 0.25)    |    0.911 |   0.907 |          0.877 |          0.912 |       0.875 |       0.911 |         nan     |        0.877 |       0.909 |  0.833 |  0.879 |
| ('D4-hetero', 0.3)     |    0.907 |   0.906 |          0.619 |          0.91  |       0.617 |       0.91  |         nan     |        0.875 |       0.911 |  0.637 |  0.789 |
| ('D4-hetero', 0.35)    |    0.899 |   0.88  |          0.531 |          0.905 |       0.53  |       0.912 |         nan     |        0.834 |       0.913 |  0.466 |  0.757 |
| ('D5-leverage', 0.25)  |    0.642 |   0.525 |          0.599 |          0.765 |       0.601 |       0.766 |           0.603 |        0.828 |       0.528 |  0.581 |  0.527 |
| ('D5-leverage', 0.3)   |    0.538 |   0.528 |          0.535 |          0.623 |       0.534 |       0.622 |           0.551 |        0.758 |       0.534 |  0.525 |  0.526 |
| ('D5-leverage', 0.35)  |    0.527 |   0.527 |          0.527 |          0.545 |       0.528 |       0.546 |           0.53  |        0.704 |       0.533 |  0.507 |  0.528 |
| ('D6-uniform', 0.25)   |    0.828 |   0.826 |          0.907 |          0.907 |       0.907 |       0.908 |           0.908 |        0.841 |       0.895 |  0.866 |  0.895 |
| ('D6-uniform', 0.3)    |    0.684 |   0.646 |          0.891 |          0.908 |       0.893 |       0.912 |           0.904 |        0.818 |       0.894 |  0.674 |  0.859 |
| ('D6-uniform', 0.35)   |    0.585 |   0.559 |          0.778 |          0.898 |       0.794 |       0.903 |           0.804 |        0.718 |       0.884 |  0.694 |  0.827 |
| ('D7-p3', 0.25)        |    0.91  |   0.902 |          0.806 |          0.91  |       0.805 |       0.907 |         nan     |        0.799 |       0.908 |  0.885 |  0.815 |
| ('D7-p3', 0.3)         |    0.906 |   0.867 |          0.564 |          0.909 |       0.564 |       0.905 |         nan     |        0.771 |       0.908 |  0.775 |  0.68  |
| ('D7-p3', 0.35)        |    0.877 |   0.748 |          0.53  |          0.856 |       0.53  |       0.858 |         nan     |        0.683 |       0.892 |  0.545 |  0.633 |
| ('D8-K2par-sep', 0.25) |    0.999 |   0.999 |          0.937 |          0.999 |       0.937 |       0.999 |         nan     |        0.833 |       0.877 |  0.682 |  0.783 |
| ('D8-K2par-sep', 0.3)  |    0.999 |   0.999 |          0.582 |          0.999 |       0.582 |       0.999 |         nan     |        0.801 |       0.91  |  0.697 |  0.779 |
| ('D8-K2par-sep', 0.35) |    0.999 |   0.998 |          0.528 |          0.999 |       0.528 |       0.999 |         nan     |        0.813 |       0.859 |  0.655 |  0.753 |