#pragma once
#include <vector>
#include "util/host_param_view.h"
#include "util/markers.h"
#include "util/logger.h"
#include <ctime>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(BenchmarkData *benchmarkData, const std::vector<HostParamView>& params, int step_index) = 0;
};

