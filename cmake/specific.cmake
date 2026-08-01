find_package(fmt CONFIG QUIET)
if(fmt_FOUND)
  message(STATUS "Found system fmt: ${fmt_DIR}")
  # Tell CPM that fmt is already handled (CPM checks CPM_PACKAGES list). Write the CACHE
  # variable directly: list(APPEND ...) creates a normal-variable shadow that does not
  # propagate into FetchContent subdirectory scopes.
  if(NOT fmt IN_LIST CPM_PACKAGES)
    set(CPM_PACKAGES "${CPM_PACKAGES};fmt" CACHE INTERNAL "" FORCE)
  endif()
else()
  CPMAddPackage(
    NAME fmt
    GIT_TAG 12.1.0
    GITHUB_REPOSITORY fmtlib/fmt
    OPTIONS "FMT_INSTALL YES" # create an installable target
  )
endif()

# When fmt is from system, tell spdlog to use it externally to avoid
# its bundled fmt conflicting with the installed fmt::fmt targets.
if(fmt_FOUND)
  set(SPDLOG_FMT_EXTERNAL YES)
endif()
CPMAddPackage(
  NAME spdlog
  GIT_TAG v1.17.0
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS "SPDLOG_INSTALL YES"
          "SPDLOG_FMT_EXTERNAL ${SPDLOG_FMT_EXTERNAL}"
)

set(SPECIFIC_LIBS fmt::fmt spdlog::spdlog)
# remember to turn off the warnings
