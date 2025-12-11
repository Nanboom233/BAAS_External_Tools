//
// Created by Nanboom233 on 2025/12/3.
//
#pragma once
#include <string>
#include <iostream>

class OcrEngine {
public:
    OcrEngine(const std::string& config_path);

    static void process_dummy(const std::string& msg);

private:
    std::string config_path_;
};