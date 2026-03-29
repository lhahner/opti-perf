#pragma once
#include <map>
#include <ctime>
#include <iostream>
#include <string>
#include <sstream>

/**
 * @brief Identifies the severity level of a log message.
 */
enum class Marker {
    INFO,
    WARNING,
    ERROR
};

/**
 * @brief Converts a marker level into a timestamped log prefix string.
 * @param marker Marker level to format.
 * @return Formatted log prefix string.
 */
inline std::string toString(Marker marker) {
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
    std::stringstream ss;

    switch (marker) {
        case Marker::INFO:
            ss << std::asctime(localTime) << " [INFO] ";
            return ss.str();
        case Marker::WARNING:
            ss << std::asctime(localTime) << " [WARNING] ";
            return ss.str();
        case Marker::ERROR:
            ss << std::asctime(localTime) << " [ERROR] ";
            return ss.str();
        default:
            ss << std::asctime(localTime) << " [UNKNOWN] ";
            return ss.str();
    }
}
