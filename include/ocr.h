//
// Created by Nanboom233 on 2025/12/11.
//
#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <utility>


class OrtEnvHolder {
public:
    static Ort::Env& Get();
};

class OnnxSession {
public:
    OnnxSession(const std::wstring& model_path, bool use_cuda=false, int intra_threads=4);
    Ort::Session& session() { return session_; }
    Ort::Session& session() const { return session_; }
    const std::vector<std::string>& input_names()  const { return input_names_; }
    const std::vector<std::string>& output_names() const { return output_names_; }
    const std::vector<std::vector<int64_t>>& input_shapes() const { return input_shapes_; }

private:
    mutable Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_names_, output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
};

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