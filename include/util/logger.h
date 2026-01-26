#include <iostream>
#include <fstream>
#include <vector>
#include "util/markers.h"

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
     */
    void logToCsv(const std::vector<std::string> &headers, const std::vector<std::string> &values, const char *filename);

private:
    // File path to store log files
    const char *filePath = "data/logs/";
};