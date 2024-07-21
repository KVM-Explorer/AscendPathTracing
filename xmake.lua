
set_languages("c++17")
set_plat("linux")
set_targetdir("build")


includes("toolchains/cpu")
includes("toolchains/npu")

target("render")
    set_kind("binary")
    add_files("src/*.cpp")
    set_toolchains("cpu")