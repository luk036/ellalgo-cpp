CPMAddPackage(
  NAME fmt
  GIT_TAG 10.2.1
  GITHUB_REPOSITORY fmtlib/fmt
  OPTIONS "FMT_INSTALL YES" # create an installable target
)

CPMAddPackage(
  NAME spdlog
  GIT_TAG v1.12.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL YES" # create an installable target
)

set(SPECIFIC_LIBS fmt::fmt spdlog::spdlog)
# remember to turn off the warnings
