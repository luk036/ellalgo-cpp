#include <ellalgo/logger.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>

namespace ellalgo {

void log_with_spdlog(const std::string& message) {
    // Create a file logger
    auto logger = spdlog::basic_logger_mt("file_logger", "ellalgo.log");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    
    spdlog::info("EllAlgo message: {}", message);
    spdlog::flush_on(spdlog::level::info);
}

} // namespace ellalgo