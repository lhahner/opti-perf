#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include "util/markers.h"
#include "util/benchmark_data.h"
#define LOG_TO_FILE_SUCESS 0
#define LOG_TO_FILE_FAILURE -1

/**
 * @brief Basic Logger wrapper class to log messages with different marker levels.
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
     * @brief Logger function to log data into a CSV file
     *
     * @param headers vector of column headers
     * @param values vector of values corresponding to the headers
     * @param filename name of the CSV file to log data into
     * 
     */
    int logToCsv(BenchmarkData benchmarkData, const char *filename);

private:
    BenchmarkData benchmarkData;
};
