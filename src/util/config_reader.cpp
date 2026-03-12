#include "util/config_reader.h"

YAML::Node ConfigReader::get_runtime_config() {
    if (config["runtime"]) {
        return config["runtime"];
    } else {
        throw std::runtime_error("Runtime configuration not found in the config file.");
    }
}

YAML::Node ConfigReader::get_optimizer_config() {
    if (config["optimizer"]) {
        return config["optimizer"];
    } else {
        throw std::runtime_error("Optimizer configuration not found in the config file.");
    }
}

YAML::Node ConfigReader::get_workload_config() {
    if (config["workload"]) {
        return config["workload"];
    } else {
        throw std::runtime_error("Workload configuration not found in the config file.");
    }
}