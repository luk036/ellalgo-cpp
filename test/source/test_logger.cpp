/**
 * @file test_logger.cpp
 * @brief Tests for the spdlogger integration
 *
 * This file contains comprehensive tests for the logger functionality,
 * including wrapper function tests and direct spdlog usage tests.
 */

#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <ellalgo/logger.hpp>
#include <fstream>
#include <iostream>
#include <string>

TEST_CASE("test_logger_wrapper_function") {
    // Test the wrapper function
    ellalgo::log_with_spdlog("Test message from wrapper function");

    // Verify the log file was created
    std::ifstream logfile("ellalgo.log");
    REQUIRE(logfile.is_open());

    // Check that the file contains the expected message
    std::string line;
    bool found = false;
    while (std::getline(logfile, line)) {
        if (line.find("Test message from wrapper function") != std::string::npos) {
            found = true;
            break;
        }
    }
    REQUIRE(found);
    logfile.close();
}

TEST_CASE("test_logger_multiple_calls") {
    // Test multiple calls to the wrapper function
    ellalgo::log_with_spdlog("First message");
    ellalgo::log_with_spdlog("Second message");
    ellalgo::log_with_spdlog("Third message");

    // Verify all messages are logged
    std::ifstream logfile("ellalgo.log");
    REQUIRE(logfile.is_open());

    std::string content((std::istreambuf_iterator<char>(logfile)),
                        std::istreambuf_iterator<char>());
    logfile.close();

    REQUIRE(content.find("First message") != std::string::npos);
    REQUIRE(content.find("Second message") != std::string::npos);
    REQUIRE(content.find("Third message") != std::string::npos);
}

TEST_CASE("test_logger_formatting") {
    // Test that the log format includes timestamp and level
    ellalgo::log_with_spdlog("Format test message");

    std::ifstream logfile("ellalgo.log");
    REQUIRE(logfile.is_open());

    std::string line;
    bool found = false;
    while (std::getline(logfile, line)) {
        if (line.find("Format test message") != std::string::npos) {
            // Check for timestamp pattern [YYYY-MM-DD HH:MM:SS.mmm]
            found = (line.find("[") == 0 && line.find("]") > 0);
            // Check for logger name
            found = found && (line.find("[file_logger]") != std::string::npos);
            // Check for log level
            found = found && (line.find("[info]") != std::string::npos);
            break;
        }
    }
    REQUIRE(found);
    logfile.close();
}

TEST_CASE("test_direct_spdlog_usage") {
    // Test direct spdlog usage for comparison
    try {
        spdlog::drop("direct_test_logger");
    } catch (...) {
    }

    auto logger = spdlog::basic_logger_mt("direct_test_logger", "direct_test.log");
    logger->set_level(spdlog::level::info);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
    logger->flush_on(spdlog::level::info);
    logger->info("Direct spdlog test message");
    logger->flush();

    // Verify the direct log file
    std::ifstream logfile("direct_test.log");
    REQUIRE(logfile.is_open());

    std::string line;
    bool found = false;
    while (std::getline(logfile, line)) {
        if (line.find("Direct spdlog test message") != std::string::npos) {
            found = true;
            break;
        }
    }
    REQUIRE(found);
    logfile.close();
}

TEST_CASE("test_logger_with_algorithm_context") {
    // Test logging in an algorithm context
    ellalgo::log_with_spdlog("Starting ellipsoid algorithm test");

    // Simulate some algorithm steps
    for (int i = 0; i < 3; ++i) {
        ellalgo::log_with_spdlog("Iteration " + std::to_string(i) + " completed");
    }

    ellalgo::log_with_spdlog("Ellipsoid algorithm test completed");

    // Verify all iteration messages are logged
    std::ifstream logfile("ellalgo.log");
    REQUIRE(logfile.is_open());

    std::string content((std::istreambuf_iterator<char>(logfile)),
                        std::istreambuf_iterator<char>());
    logfile.close();

    REQUIRE(content.find("Starting ellipsoid algorithm test") != std::string::npos);
    REQUIRE(content.find("Iteration 0 completed") != std::string::npos);
    REQUIRE(content.find("Iteration 1 completed") != std::string::npos);
    REQUIRE(content.find("Iteration 2 completed") != std::string::npos);
    REQUIRE(content.find("Ellipsoid algorithm test completed") != std::string::npos);
}

TEST_CASE("test_logger_error_handling") {
    // Test that the logger handles errors gracefully
    // This should not throw an exception
    REQUIRE_NOTHROW(ellalgo::log_with_spdlog("Error handling test message"));

    // The logger should complete without throwing an exception
    // The actual log file verification is handled by other tests
    // This test focuses on error handling and not throwing exceptions
}