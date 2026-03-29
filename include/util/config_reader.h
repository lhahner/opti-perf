#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <exception>

/**
 * @brief Reads and exposes structured configuration sections from a YAML file.
 */
class ConfigReader {
    private:
        std::string config_file_path;
        YAML::Node config;

    public:
        /**
         * @brief Loads the YAML configuration file from disk.
         * @param path Path to the configuration file.
         */
        ConfigReader(const std::string& path) : config_file_path(path) {
            try {
                config = YAML::LoadFile(config_file_path);
            } catch (const std::exception& exception) {
                std::cerr << "Error loading config file: " << exception.what() << std::endl;
                throw;
            }
        }
           
        /**
         * @brief Returns the runtime configuration section.
         * @return YAML node containing runtime settings.
         */
        YAML::Node get_runtime_config(); 
        
        /**
         * @brief Returns the optimizer configuration section.
         * @return YAML node containing optimizer settings.
         */
        YAML::Node get_optimizer_config();
        
        /**
         * @brief Returns the workload configuration section.
         * @return YAML node containing workload settings.
         */
        YAML::Node get_workload_config();
    };
