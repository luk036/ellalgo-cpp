/**
 * @file logger.cpp
 * @brief Implementation of logging utilities using spdlog
 *
 * This file implements the logging functionality for the ellalgo library
 * using the spdlog library as the backend.
 */

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <ellalgo/logger.hpp>
#include <memory>

namespace ellalgo {

    /**
     * @brief Log a message using spdlog to a file
     * 
     * This function creates a file logger that writes to "ellalgo.log",
     * sets it as the default logger, and logs the provided message with
     * an info level. The logger is configured to flush on info level
     * messages to ensure they are written to disk immediately.
     * 
     * @param[in] message The message to log
     */
    void log_with_spdlog(const std::string& message) {
        // Create a file logger
        auto logger = spdlog::basic_logger_mt("file_logger", "ellalgo.log");
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::info);

        spdlog::info("EllAlgo message: {}", message);
        spdlog::flush_on(spdlog::level::info);
    }

}  // namespace ellalgo