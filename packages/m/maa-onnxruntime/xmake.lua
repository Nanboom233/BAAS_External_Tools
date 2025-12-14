-- onnxruntime with DirectML, CUDA, or CoreML support xmake package definition
-- inspired by maa-onnxruntime from MaaDeps
-- @MaaAssistantArknights/MaaDeps/files/vcpkg-overlay/ports/maa-onnxruntime
-- most copied from official onnxruntime xmake package
-- @xmake-io/xmake-repo/blob/4cfbaa42adc11fcfe6c3efe9136ac2a30025c83a/packages/o/onnxruntime/xmake.lua
package("maa-onnxruntime")
    set_homepage("https://www.onnxruntime.ai")
    set_description("ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator")
    set_license("MIT")

    add_urls("https://github.com/microsoft/onnxruntime.git",
             { version = "1.19.2", branch = "v1.19.2" })

    add_versions("1.19.2", "26250ae74d2c9a3c6860625ba4a147ddfb936907")

    -- feature-like options
    add_configs("cuda",      {description = "Build with CUDA support", default = false, type = "boolean"})
    add_configs("directml",  {description = "Build with DirectML support", default = false, type = "boolean"})
--     add_configs("coreml",    {description = "Build with CoreML support", default = false, type = "boolean"})
    add_configs("shared", {description = "Download shared binaries.", default = true, type = "boolean", readonly = true})

    on_load(function (package)
        package:add("deps", "cmake")
        if package:config("cuda") then
            package:add("deps", "cuda", {configs = {utils = {"cudart", "nvrtc"}}})
        end
        if package:is_plat("windows") and package:config("directml") then
            package:add("deps", "directml-bin")
            package:add("deps", "directx-headers")
        end
--             if package:is_plat("macosx") and package:config("coreml") then
--                 -- TODO: add coreml support
--             end
    end)

    on_install("windows", "linux", "macosx", function (package)
        import("package.tools.cmake")

        local configs = {}

        table.insert(configs, "-Donnxruntime_USE_VCPKG=OFF")
        table.insert(configs, "-Donnxruntime_BUILD_WEBASSEMBLY=OFF")
        table.insert(configs, "-Donnxruntime_ENABLE_PYTHON=OFF")
        table.insert(configs, "-Donnxruntime_ENABLE_TRAINING=OFF")
        table.insert(configs, "-Donnxruntime_ENABLE_TRAINING_APIS=OFF")
        table.insert(configs, "-Donnxruntime_ENABLE_MIMALLOC=OFF")
        table.insert(configs, "-Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF")
        table.insert(configs, "-Donnxruntime_BUILD_UNIT_TESTS=OFF")

        -- cuda / directml / coreml feature options
        if package:config("cuda") then
            table.insert(configs, "-Donnxruntime_USE_CUDA=ON")
            table.insert(configs, "-Donnxruntime_USE_CUDA_NHWC_OPS=ON")
        end
        if package:config("directml") then
            table.insert(configs, "-Donnxruntime_USE_DML=ON")
            table.insert(configs, "-Donnxruntime_USE_CUSTOM_DIRECTML=ON")
        end
        if package:config("coreml") then
            table.insert(configs, "-Donnxruntime_USE_COREML=ON")
        end

        -- other options
        if package:config("shared") then
            table.insert(configs, "-Donnxruntime_BUILD_SHARED_LIB=ON")
        end

        cmake.install(package, configs, {buildir = "build"})

        -- copy required dlls to bin folder on windows
        if package:is_plat("windows") then
            os.mv("lib/*.dll", package:installdir("bin"))
        end
        os.cp("*", package:installdir())
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            #include <array>
            #include <cstdint>
            void test() {
                std::array<float, 2> data = {0.0f, 0.0f};
                std::array<int64_t, 1> shape{2};

                Ort::Env env;

                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                auto tensor = Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), shape.data(), shape.size());
            }
        ]]}, {configs = {languages = "c++17"}, includes = "onnxruntime_cxx_api.h"}))
    end)
