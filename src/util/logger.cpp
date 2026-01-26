#include "util/logger.h"

void Logger::log(Marker marker, const char *message)
{
    std::cout << toString(marker) << message << std::endl;
}

void Logger::logToCsv(const std::vector<std::string>& headers, const std::vector<std::string>& values, const char* filename) {
    std::ofstream file;
    file.open(std::string(filePath) + filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << toString(Marker::ERROR) << "Failed to open CSV file: " << filename << std::endl;
        return;
    }

    if (file.tellp() == 0) {
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i < headers.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    for (size_t i = 0; i < values.size(); ++i) {
        file << values[i];
        if (i < values.size() - 1) {
            file << ",";
        }
    }
    file << "\n";
    file.close();
} 
