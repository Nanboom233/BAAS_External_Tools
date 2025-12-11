//
// Created by Nanboom233 on 2025/12/12.
//
#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "onnx_session.h"


struct TextBox {
    std::vector<cv::Point2f> quad; // 四点
    float score{0.f};
};

class Detector {
public:
    struct Params {
        int   max_size   = 1280;
        float bin_thresh = 0.3f;
        float box_thresh = 0.5f;
        float unclip_ratio = 1.8f;
        int   min_size   = 3;
    };

    Detector(const std::wstring& det_model, bool use_cuda=false, const Params& p={});
    // 本阶段先实现：预处理 + 前向，返回 prob 图；后处理下一阶段补
    cv::Mat forward_prob(const cv::Mat& bgr, cv::Size& net_size_out, float& scale_out) const;
    std::vector<TextBox> detect(const cv::Mat& bgr) const;

private:
    OnnxSession det_session_;
    Params params_;
    static cv::Mat resize_to_h32(const cv::Mat& bgr,
                                           int limit_side_len,
                                           int stride,
                                           float* out_ratio_h,
                                           float* out_ratio_w);
    static void normalize_rgb(cv::Mat& rgb);
    static std::vector<TextBox> decode_boxes(const cv::Mat &prob, float bin_th, float box_th, int min_size,
                                            const cv::Size &content_size, const cv::Size &ori_size, float unclip_ratio, float down);
};
