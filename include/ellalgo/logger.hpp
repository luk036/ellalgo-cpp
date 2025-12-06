/**
 * @file logger.hpp
 * @brief Logging utilities for the ellalgo library
 *
 * This header provides logging functionality using spdlog as the backend.
 * It offers a simple interface for logging messages throughout the library.
 */

#pragma once

#include <string>

namespace ellalgo {

    /**
     * @brief Log a message using spdlog
     *
     * This function logs a message using the spdlog library. It provides
     * a simple interface for logging messages throughout the ellalgo
     * library without exposing the full spdlog API.
     *
     * @param[in] message The message to log
     */
    void log_with_spdlog(const std::string& message);

}