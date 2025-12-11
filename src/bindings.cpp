//
// Created by Nanboom233 on 2025/12/3.
//
#include <pybind11/pybind11.h>
#include "ocr_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(baas_external_tools, m) {
    m.doc() = "External C++ OCR Core";

    py::class_<OcrEngine>(m, "OcrEngine")
        .def(py::init<const std::string&>())
        .def_static("process_dummy", &OcrEngine::process_dummy);
}

