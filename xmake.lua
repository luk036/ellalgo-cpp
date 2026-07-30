add_rules("mode.debug", "mode.release")
set_languages("c++20")

if is_plat("windows") then
    add_cxflags("/utf-8", "/W4", "/WX", "/wd4819")
else
    add_cxflags("-Wall", "-Wextra", "-Wpedantic", "-Werror")
end

-- mode.coverage adds --coverage which only GCC/Clang support
if not is_plat("windows") then
    add_rules("mode.coverage")
end

add_requires("doctest")
add_requires("nanobench")
add_requires("spdlog")
add_requires("cxxopts")

target("ellalgo")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_files("source/*.cpp")
    add_packages("spdlog")

target("test_all")
    set_kind("binary")
    add_deps("ellalgo")
    add_files("test/source/*.cpp")
    add_includedirs("include")
    add_packages("doctest", "spdlog")
    add_tests("default")

-- ---------------------------------------------------------------------------
-- Benchmarks
-- ---------------------------------------------------------------------------

option("benchmarks")
    set_default(false)
    set_showmenu(true)
    set_description("Build benchmarks")

if has_config("benchmarks") then
    for _, bench_file in ipairs(os.files("bench/*.cpp")) do
        local name = path.basename(bench_file)
        target("bench_" .. name)
            set_kind("binary")
            add_deps("ellalgo")
            add_files(bench_file)
            add_includedirs("include")
            add_packages("nanobench")
    end
end

-- ---------------------------------------------------------------------------
-- Standalone
-- ---------------------------------------------------------------------------

target("ellalgo_standalone")
    set_kind("binary")
    add_deps("ellalgo")
    add_files("standalone/source/main.cpp")
    add_packages("cxxopts", "spdlog")
