#ifndef INCLUDE_UTIL_HOST_PARAM_VIEW_H_
#define INCLUDE_UTIL_HOST_PARAM_VIEW_H_

#pragma once
#include <vector>
#include <cstddef>

struct HostParamView {
    float* data;      // parameter buffer (e.g., W)
    float* grad;      // gradient buffer (e.g., dW)
    size_t count;     // number of float elements
    const char* name; // optional
};

#endif /* INCLUDE_UTIL_HOST_PARAM_VIEW_H_ */
