#include "util/logger.h"

#include <filesystem>

void Logger::log(Marker marker, const char *message)
{
    std::cout << toString(marker) << message << std::endl;
}

int Logger::logToCsv(BenchmarkData benchmarkData, const char *filename)
{
    namespace fs = std::filesystem;
    fs::path out_path(filename);

    if (!out_path.is_absolute())
    {
#ifdef OPTI_PERF_SOURCE_DIR
        fs::path base(OPTI_PERF_SOURCE_DIR);
#else
        fs::path base = fs::current_path();
#endif
        out_path = base / "data" / "logs" / out_path;
    }

    std::ofstream file;
    file.open(out_path, std::ios::app);
    if (!file.is_open())
    {
        std::cerr << toString(Marker::ERROR) << "Failed to open CSV file: " << out_path.string() << std::endl;
        return LOG_TO_FILE_FAILURE;
    }

    if (file.tellp() == 0)
    {
        file << BenchmarkData::csv_header() << "\n"; // Write header only once
    }
    file << benchmarkData.to_csv_string() << "\n";
    file.close();
    return LOG_TO_FILE_SUCESS;
}
