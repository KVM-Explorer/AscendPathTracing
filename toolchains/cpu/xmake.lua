toolchain("cpu")

    set_kind("standalone")

    set_toolset("cc", "gcc" )
    set_toolset("cxx", "gcc", "g++")
    set_toolset("ld", "g++", "gcc")
    set_toolset("sh", "g++", "gcc")
    set_toolset("ar", "ar")
    set_toolset("strip", "strip")
    set_toolset("objcopy", "objcopy")
    set_toolset("ranlib", "ranlib")
    set_toolset("mm", "gcc")
    set_toolset("mxx", "gcc", "g++")
    set_toolset("as", "gcc")

   
    on_check(function (toolchain)
        return import("lib.detect.find_tool")("gcc")
    end)

    on_load(function (toolchain)
        local gcc = toolchain:find_tool("gcc")
        if gcc then
            toolchain:add("cxflags", "-g")
        end

        local cmake = toolchain:find_package("cmake")
        if cmake then
            local tikicpulib_name = "tikicpulib::" .. _OPTIONS["ASCEND_PRODUCT_TYPE"]
            if is_plat("windows") then
                tikicpulib_name = tikicpulib_name .. ".lib"
            else
                tikicpulib_name = "-l" .. tikicpulib_name
            end
            toolchain:add("links", tikicpulib_name)
        end

        toolchain:add("links", "ascendcl")
        toolchain:add("includedirs", "${ASCEND_INSTALL_PATH}/acllib/include")
        toolchain:add("linkdirs", "${ASCEND_INSTALL_PATH}/tools/tikicpulib/lib")

        toolchain:add("defines", "_GLIBCXX_USE_CXX11_ABI=0")

        toolchain:add("configmap", "ASCEND_PRODUCT_TYPE", "Ascend910A")
        toolchain:add("configmap", "ASCEND_RUN_MODE", "cpu")
        toolchain:add("configmap", "ASCEND_INSTALL_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

        toolchain:add("build_command", "prebuild", function (target)
            os.exec("rm -rf build && mkdir build")
        end)
    end)
toolchain_end()