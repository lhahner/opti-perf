#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include "util/markers.h"
#include "util/benchmark_data.h"
#define LOG_TO_FILE_SUCESS 0
#define LOG_TO_FILE_FAILURE -1

/**
 * @brief Logs diagnostic messages and benchmark rows.
 */
class Logger
{
public:
    /**
     * @brief Logs a message with a specified marker level.
     *
     * @param marker The marker level (INFO, WARNING, ERROR).
     * @param message The message to log.
     */
    void log(Marker marker, const char *message);

    /**
     * @brief Appends benchmark data to a CSV log file.
     * @param benchmarkData Benchmark data row to serialize.
     * @param filename Output CSV file name or path.
     * @return `LOG_TO_FILE_SUCESS` on success, otherwise `LOG_TO_FILE_FAILURE`.
     */
    int logToCsv(BenchmarkData benchmarkData, const char *filename);

private:
    BenchmarkData benchmarkData;
};
