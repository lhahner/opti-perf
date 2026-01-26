#pragma once
#include <map>

enum class Marker {
    INFO,
    WARNING,
    ERROR
};

inline const char* toString(Marker marker) {
    switch (marker) {
        case Marker::INFO:
            return "[INFO] ";
        case Marker::WARNING:
            return "[WARNING] ";
        case Marker::ERROR:
            return "[ERROR] ";
        default:
            return "[UNKNOWN] ";
    }
}