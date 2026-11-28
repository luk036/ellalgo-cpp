set_languages("c++20")

add_rules("mode.debug", "mode.release", "mode.coverage")
add_requires("doctest", "fmt", "nanobench", "spdlog")
add_requires("cxxopts", {optional = true})

add_includedirs("include")

if is_mode("coverage") then
    add_cxflags("-ftest-coverage", "-fprofile-arcs", {force = true})
end

if is_plat("linux") then
    set_warnings("all", "error")
    -- Termux/Android ships its libc++ headers under $PREFIX
    local prefix = os.getenv("PREFIX")
    if prefix then
        add_cxflags("-Wno-unused-command-line-argument", {force = true})
        add_sysincludedirs(prefix .. "/include/c++/v1", prefix .. "/include", {public = true})
    end
    -- host-native tuning for auto-vectorization
    if is_mode("release") then
        add_cxflags("-march=native", "-mtune=native", {force = true})
    end
elseif is_plat("windows") then
    add_cxflags("/EHsc /utf-8 /W4 /WX /wd4702", {force = true})
    if is_mode("release") then
        add_cxflags("/arch:AVX2", {force = true})
    end
end

-- dependency-free cutting-plane framework (Strategy-pattern Context)
target("EllAlgoCore")
    set_kind("headeronly")
    add_headerfiles("include/(ellalgo/ell_config.hpp)",
                    "include/(ellalgo/cutting_plane.hpp)",
                    "include/(ellalgo/half_nonnegative.hpp)")

target("EllAlgo")
    set_kind("static")
    add_deps("EllAlgoCore")
    add_files("source/*.cpp")
    add_packages("fmt", "spdlog")

target("test_ellalgo")
    add_deps("EllAlgo")
    add_files("test/source/*.cpp")
    add_packages("doctest", "fmt", "spdlog")
    add_tests("default")

-- nanobench micro-benchmarks
target("test_ell")
    add_deps("EllAlgo")
    add_files("bench/BM_ell.cpp")
    add_packages("nanobench")

target("test_ell_compare")
    add_deps("EllAlgo")
    add_files("bench/BM_ell_compare.cpp")
    add_packages("nanobench")

target("test_lmi")
    add_deps("EllAlgo")
    add_files("bench/BM_lmi.cpp")
    add_packages("nanobench")

target("test_lowpass")
    add_deps("EllAlgo")
    add_files("bench/BM_lowpass.cpp")
    add_packages("nanobench")

target("EllAlgoStandalone")
    add_deps("EllAlgo")
    add_files("standalone/source/main.cpp")
    add_packages("cxxopts", "spdlog")
    set_filename("EllAlgo")
