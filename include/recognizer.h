//
// Created by Nanboom233 on 2025/12/12.
//

#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
// 改为只依赖会话头，避免循环
#include "onnx_session.h"

struct RecResult { std::string text; float score{0.f}; };

class Recognizer {
public:
    struct Params {
        int imgH=48;
        int imgW=960;
        bool insert_space=true;  // 启用“空格插入”启发式
        int  min_space_frames=2; // 判定空格的最小时间步
        float space_prob=0.00000f;   // 判定空格的概率阈值
        bool  fallback_space_from_blank=true;
        int   min_blank_frames_for_space=3;
        int   max_imgW=2304;
        float tile_overlap=0.25f;
        int   merge_overlap_chars=6;
    }; // 最大宽
    Recognizer(const std::wstring& rec_model, const std::string& keys_path,
               bool use_cuda=false, const Params& p={});
    RecResult infer(const cv::Mat& crop) const;

private:
    int space_k_ = -1; // 模型里的“空格”类别索引（1..C-1），-1 表示 keys 无空格
    OnnxSession rec_;
    Params p_;
    std::vector<std::string> charset_;
    static void normalize_rec(cv::Mat& img);
    std::string ctc_greedy(const cv::Mat& logits_TxC, float* avg_conf=nullptr) const;
};