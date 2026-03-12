#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <exception>

/**
 * @brief A class for reading configuration files.
 * Handling hyperparemeter loading and parsing for the runtime, optimizer, and workload components.
 */
class ConfigReader {
    private:
        std::string config_file_path;
        YAML::Node config;

    public:
        ConfigReader(const std::string& path) : config_file_path(path) {
            try {
                config = YAML::LoadFile(config_file_path);
            } catch (const std::exception& exception) {
                std::cerr << "Error loading config file: " << exception.what() << std::endl;
                throw;
            }
        }
           
        /**
         * @brief Get the runtime config object,
         * which contains optimizer type, workload and frameworkd 
         * to use.
         */
        YAML::Node get_runtime_config(); 
        
        /**
         * @brief Get the optimizer config object,
         * which contains the hyperparameters for the optimizer, 
         * such as learning rate, batch size, etc.
         */
        YAML::Node get_optimizer_config();
        
        /**
         * @brief Get the workload config object, describing
         * the workload size and others.
         */
        YAML::Node get_workload_config();
    };
