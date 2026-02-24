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

namespace ellalgo {

    /**
     * @brief Log a message using spdlog to a file
     *
     * This function creates a file logger that writes to "ellalgo.log",
     * logs the provided message with an info level, and immediately flushes
     * to ensure the message is written to disk. The logger is configured
     * with a custom pattern and always creates a fresh instance to ensure
     * proper file handling.
     *
     * @param[in] message The message to log
     */
    void log_with_spdlog(const std::string& message) {
        // Always create a fresh logger to ensure proper file handling
        std::shared_ptr<spdlog::logger> logger;
        try {
            // Try to drop the existing logger first
            spdlog::drop("file_logger");
        } catch (...) {
            // Ignore if logger doesn't exist
        }

        // Create a new logger
        logger = spdlog::basic_logger_mt("file_logger", "ellalgo.log");
        if (logger) {
            logger->set_level(spdlog::level::info);
            logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
            logger->flush_on(spdlog::level::info);
            logger->info("EllAlgo message: {}", message);
            logger->flush();
        }
    }

}  // namespace ellalgo