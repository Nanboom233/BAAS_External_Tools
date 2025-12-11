#include "detecter.h"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <numeric>

Detector::Detector(const std::wstring &det_model, bool use_cuda, const Params &p) :
    det_session_(det_model, use_cuda), params_(p) {}

// 归一化：在 RGB 空间做 (x/255 - mean) / std
void Detector::normalize_rgb(cv::Mat &rgb) {
    CV_Assert(rgb.type() == CV_32FC3);
    // mean/std for RGB
    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar stdv(0.229, 0.224, 0.225);
    cv::subtract(rgb, mean, rgb);
    cv::divide(rgb, stdv, rgb);
}

// 检测前处理：等比缩放 + pad 到 32 的倍数
cv::Mat Detector::resize_to_h32(const cv::Mat &bgr, int limit_side_len = 960, int stride = 32,
                                float *out_ratio_h = nullptr, float *out_ratio_w = nullptr) {
    const int h = bgr.rows, w = bgr.cols;
    float ratio = 1.f;
    if (limit_side_len > 0) {
        float r_h = limit_side_len / static_cast<float>(h);
        float r_w = limit_side_len / static_cast<float>(w);
        // 以“max”策略为例：限制长边
        ratio = std::min(1.f, std::min(r_h, r_w)); // 长边>limit才缩小
    }
    int nh = std::max(1, static_cast<int>(std::round(h * ratio)));
    int nw = std::max(1, static_cast<int>(std::round(w * ratio)));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    // pad 到 stride 的倍数（通常 stride=32）
    int ph = (nh + stride - 1) / stride * stride;
    int pw = (nw + stride - 1) / stride * stride;
    if (out_ratio_h)
        *out_ratio_h = static_cast<float>(nh) / static_cast<float>(h);
    if (out_ratio_w)
        *out_ratio_w = static_cast<float>(nw) / static_cast<float>(w);

    cv::Mat padded(ph, pw, bgr.type(), cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(0, 0, nw, nh)));
    return padded;
}


cv::Mat Detector::forward_prob(const cv::Mat &bgr, cv::Size &content_size_out, float &down_out) const {
    float r_h = 1.f, r_w = 1.f;
    cv::Mat img = resize_to_h32(bgr, /*limit_side_len=*/params_.max_size, /*stride=*/32, &r_h, &r_w);

    // 记录 resize 后的有效尺寸（未 pad）
    int nh = std::max(1, int(std::round(bgr.rows * r_h)));
    int nw = std::max(1, int(std::round(bgr.cols * r_w)));
    content_size_out = cv::Size(nw, nh);

    // 预处理…
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    normalize_rgb(rgb);

    // NCHW
    cv::Mat chw;
    cv::dnn::blobFromImage(rgb, chw, 1.0, cv::Size(), cv::Scalar(), false, false);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> ishape{1, 3, (int64_t) rgb.rows, (int64_t) rgb.cols};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, (float *) chw.data, (size_t) chw.total(), ishape.data(),
                                                       ishape.size());

    auto in_names = det_session_.input_names();
    auto out_names = det_session_.output_names();
    std::vector<const char *> in{in_names[0].c_str()}, out{out_names[0].c_str()};
    auto outputs = det_session_.session().Run(Ort::RunOptions{nullptr}, in.data(), &input, 1, out.data(), 1);

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    auto oshape = info.GetShape();
    int oh = (int) oshape[oshape.size() == 4 ? 2 : oshape.size() == 3 ? 1 : 0];
    int ow = (int) oshape[oshape.size() == 4 ? 3 : oshape.size() == 3 ? 2 : 1];
    const float *ptr = outputs[0].GetTensorData<float>();

    // 推断下采样步长 d（通常=4）
    down_out = std::max(1.f, std::round((float) rgb.rows / (float) oh));

    cv::Mat prob(oh, ow, CV_32F, const_cast<float *>(ptr));
    cv::Mat prob_copy = prob.clone();
    double mn, mx;
    cv::minMaxLoc(prob_copy, &mn, &mx);
    if (mx > 1.5 || mn < -0.5) {
        cv::Mat neg;
        cv::exp(-prob_copy, neg);
        prob_copy = 1.0f / (1.0f + neg);
    }
    return prob_copy;
}

// rect 外扩：保持中心与角度不变，w/h 增加 2d，d = area*unclip_ratio/perimeter
static inline cv::RotatedRect expand_rect(const cv::RotatedRect &r, float unclip_ratio) {
    if (unclip_ratio <= 1.f)
        return r;
    float w = r.size.width, h = r.size.height;
    float area = std::max(1.f, w * h);
    float per = std::max(1e-3f, 2.f * (w + h));
    float d = area * unclip_ratio / per; // 注意：这是额外增加的边沿厚度
    cv::RotatedRect out = r;
    out.size.width = std::max(1.f, w + 2.f * d);
    out.size.height = std::max(1.f, h + 2.f * d);
    return out;
}

std::vector<TextBox> Detector::decode_boxes(const cv::Mat &prob, float bin_th, float box_th, int min_size,
                                            const cv::Size &content_size, const cv::Size &ori_size, float unclip_ratio,
                                            float down) {
    const int nw = content_size.width, nh = content_size.height;
    const float sx = (float) ori_size.width / (float) nw; // 有效内容->原图
    const float sy = (float) ori_size.height / (float) nh;

    // 1) 二值化 + 轻微膨胀，减少断裂
    cv::Mat bin;
    cv::threshold(prob, bin, bin_th, 255, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U);
    {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
        cv::dilate(bin, bin, k, cv::Point(-1, -1), 1);
    }

    // 2) 轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<TextBox> boxes;
    boxes.reserve(contours.size());

    int kept_after_size = 0, kept_after_score = 0;

    for (auto &c: contours) {
        if (c.size() < 4)
            continue;

        // 原始最小外接矩形
        cv::RotatedRect rr0 = cv::minAreaRect(c);
        if (std::min(rr0.size.width, rr0.size.height) < min_size)
            continue;
        kept_after_size++;

        // 用“原始区域”算分数（不要用外扩后的区域来算分，会稀释分数）
        // 这里用矩形四点近似掩膜，或直接用原始轮廓 c（更贴合）
        float score = 0.f;
        {
            cv::Mat mask = cv::Mat::zeros(prob.size(), CV_8U);
            // 更贴合：直接用原始轮廓
            std::vector<std::vector<cv::Point>> polys{c};
            cv::fillPoly(mask, polys, 255);
            score = (float) cv::mean(prob, mask)[0];
        }
        if (score < box_th)
            continue;
        kept_after_score++;

        // 通过阈值后再做外扩，得到用于裁剪的四点
        cv::RotatedRect rr = expand_rect(rr0, unclip_ratio);

        cv::Point2f pts[4];
        rr.points(pts);

        TextBox tb;
        tb.score = score;
        tb.quad.reserve(4);
        for (int i = 0; i < 4; ++i) {
            float x = pts[i].x * sx, y = pts[i].y * sy;
            x = std::min(std::max(0.f, x), float(ori_size.width - 1));
            y = std::min(std::max(0.f, y), float(ori_size.height - 1));
            tb.quad.emplace_back(x, y);
        }
        boxes.emplace_back(std::move(tb));
    }

#ifndef NDEBUG
    std::cout << "[DET] contours=" << contours.size() << " kept_size=" << kept_after_size
              << " kept_score=" << kept_after_score << " boxes=" << boxes.size() << "\n";
#endif
    return boxes;
}

std::vector<TextBox> Detector::detect(const cv::Mat &bgr) const {
    cv::Size content_size;
    float down = 4.f;
    cv::Mat prob = forward_prob(bgr, content_size, down);
    return decode_boxes(prob, params_.bin_thresh, params_.box_thresh, params_.min_size, content_size, bgr.size(),
                        params_.unclip_ratio, down); // 新增
}