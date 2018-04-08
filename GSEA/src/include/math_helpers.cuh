#include <cmath>

#define CUDA_GSEA_INV_LOG2 (1.44269504089)


// default math

template <class T> 
T cuabs(const T& x) {
    return x < 0 ? -x : x;
}

template <class T> 
T cumin(const T& x, const T& y) {
    return x < y ? x : y;
}

template <class T> 
T cumax(const T& x, const T& y) {
    return x < y ? y : x;
}

// specializations for uniform code


float cusqrt(const float& x) {
    return sqrtf(x);
}


double cusqrt(const double& x) {
    return sqrt(x);
}


float culog(const float& x) {
    return logf(x);
}


double culog(const double& x) {
    return log(x);
}


