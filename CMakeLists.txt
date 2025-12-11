# dependencies of pybind11
find_package(Python 3.14 COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# dependencies of ocr module
find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
        "src/*.cpp"
        "include/*.h"
)

#set(ONNXRUNTIME_ROOT "C:/Users/Nanboom233/Desktop/Code/blue_archive_auto_script/core/external_tools/external_tools/external/onnxruntime-win-x64")
#include_directories(${ONNXRUNTIME_ROOT}/include)
#link_directories(${ONNXRUNTIME_ROOT}/lib)

pybind11_add_module(baas_external_tools ${SOURCE_FILES})

target_include_directories(baas_external_tools PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/external_tools/include
)

set_target_properties(baas_external_tools PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        INTERPROCEDURAL_OPTIMIZATION ON

        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pre_build"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pre_build"
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pre_build"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/pre_build"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/pre_build"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/pre_build"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/pre_build"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/pre_build"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/pre_build"
)

target_link_libraries(baas_external_tools
        PRIVATE
        onnxruntime::onnxruntime
        ${OpenCV_LIBS}
)

add_custom_command(
        TARGET baas_external_tools POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Generating .pyi stub file..."
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:baas_external_tools>
        ${Python_EXECUTABLE} -m pybind11_stubgen baas_external_tools
        -o $<TARGET_FILE_DIR:baas_external_tools>
        --root-suffix ""
        --exit-code
        WORKING_DIRECTORY $<TARGET_FILE_DIR:baas_external_tools>
        VERBATIM
)

# test executable
add_subdirectory(tests)

install(TARGETS baas_external_tools DESTINATION core/external_tools/pre_build)
