#include "util/logger.h"

void Logger::log(Marker marker, const char *message)
{
    std::cout << toString(marker) << message << std::endl;
}

int Logger::logToCsv(BenchmarkData benchmarkData, const char *filename)
{
    std::ofstream file;
    file.open(std::string(filePath) + filename, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << toString(Marker::ERROR) << "Failed to open CSV file: " << filename << std::endl;
        return LOG_TO_FILE_FAILURE;
    }

    if (file.tellp() == 0)
    {
        file << benchmarkData.getCSVHeader() << "\n"; // Write header only once
    }
    file << benchmarkData.toCSVString() << "\n";
    file.close();
    return LOG_TO_FILE_SUCESS;
}
