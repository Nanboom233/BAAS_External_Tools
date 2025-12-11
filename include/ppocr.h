//
// Created by Nanboom233 on 2025/12/12.
//

#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "detecter.h"
#include "recognizer.h"

class PPOCR {
public:
    PPOCR(const std::wstring& det_model,
          const std::wstring& rec_model,
          const std::string&  keys_path,
          bool use_cuda=false);

    // 完整 pipeline：返回 (框, 文本)
    std::vector<std::pair<TextBox, RecResult>> run(const cv::Mat& bgr) const;

private:
    Detector det_;
    Recognizer rec_;
    static cv::Mat crop_quad_upright(const cv::Mat& img, const std::vector<cv::Point2f>& q);
};
