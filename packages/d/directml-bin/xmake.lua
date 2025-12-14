-- DirectML Binaries xmake package definition
-- inspired by directml-bin from MaaDeps
-- @MaaAssistantArknights/MaaDeps/files/vcpkg-overlay/ports/directml-bin
package("directml-bin")
    set_homepage("https://www.nuget.org/packages/Microsoft.AI.DirectML")
    set_description("DirectML (standalone) - High-performance, hardware-accelerated DirectX 12 machine learning library.")
    set_license("MIT")

    set_urls("https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$(version)", {alias = "nupkg"})

    add_versions("1.15.4", "sha256:SKIP")

    if is_plat("windows") then
        on_install(function (package)
            local arch_map = {
                ["x64"] = "x64-win",
                ["x86"] = "x86-win",
                ["arm64"] = "arm64-win"
            }

            local nuget_arch = arch_map[package:arch()]
            if not nuget_arch then
                raise("Unsupported architecture: " .. package:arch())
            end

            -- install to include/
            os.cp("include", package:installdir())

            -- install binaries
            local bin_source = path.join("bin", nuget_arch)

            -- install library
            os.cp(path.join(bin_source, "*.lib"), package:installdir("lib"))

            -- copy dlls and pdbs to bin/
            os.cp(path.join(bin_source, "*.dll"), package:installdir("bin"))
            os.cp(path.join(bin_source, "*.pdb"), package:installdir("bin"))


            -- install copyright
            os.cp("LICENSE.txt", package:installdir("share/directml-bin/copyright"))

            -- add links config
            package:add("links", "DirectML")
        end)
    end

    on_test(function (package)
        assert(package:has_cfuncs("DMLCreateDevice", {includes = "DirectML.h"}))
    end)