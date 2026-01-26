#pragma once
#include <vector>
#include "util/host_param_view.h"
#include "util/markers.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<HostParamView>& params, int step_index) = 0;
};

