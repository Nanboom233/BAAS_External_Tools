add_executable(cpp_test main.cpp)

target_sources(cpp_test PRIVATE
        ${PROJECT_SOURCE_DIR}/external_tools/src/detector.cpp
        ${PROJECT_SOURCE_DIR}/external_tools/src/recognizer.cpp
        ${PROJECT_SOURCE_DIR}/external_tools/src/ppocr.cpp
        ${PROJECT_SOURCE_DIR}/external_tools/src/onnx_session.cpp
)

#set(ONNXRUNTIME_ROOT "C:/Users/Nanboom233/Desktop/Code/blue_archive_auto_script/core/external_tools/external_tools/external/onnxruntime-win-x64")
#include_directories(${ONNXRUNTIME_ROOT}/include)
#link_directories(${ONNXRUNTIME_ROOT}/lib)

target_include_directories(cpp_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/external_tools/include
)

set_target_properties(cpp_test PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        INTERPROCEDURAL_OPTIMIZATION ON
)

target_link_libraries(cpp_test
        PRIVATE
        onnxruntime::onnxruntime
        ${OpenCV_LIBS}
)