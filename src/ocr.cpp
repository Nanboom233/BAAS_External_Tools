//
// Created by Nanboom233 on 2025/12/11.
//
#include "ocr.h"
#include <fstream>
#include <numeric>
#include <opencv2/dnn.hpp>


Ort::Env &OrtEnvHolder::Get() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "baas_ocr");
    return env; // 单例：程序全程只创建一次
}

OnnxSession::OnnxSession(const std::wstring &model_path, bool use_cuda, int intra_threads) {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(intra_threads);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 此处演示 CPU；需要 CUDA 时可按你的 ORT 包添加 Provider
    session_ = Ort::Session(OrtEnvHolder::Get(), model_path.c_str(), opt);

    /**
     * 初始化输入输出的名称和形状信息
     */
    size_t in_cnt = session_.GetInputCount();
    size_t out_cnt = session_.GetOutputCount();

    input_names_.resize(in_cnt);
    output_names_.resize(out_cnt);
    input_shapes_.resize(in_cnt);

    for (size_t i = 0; i < in_cnt; ++i) {
        auto nm = session_.GetInputNameAllocated(i, allocator_);
        input_names_[i] = nm.get(); // 拷贝字符串
        // 先检查是否为张量输入，避免非张量类型导致形状读取异常
        Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        auto shp = info.GetShape();
        // 将动态维(可能为0或-1)统一规范为-1，避免后续错误使用
        for (auto &d: shp) {
            if (d == 0) {
                d = -1;
            }
        }
        input_shapes_[i] = std::move(shp);
    }
    for (size_t i = 0; i < out_cnt; ++i) {
        auto nm = session_.GetOutputNameAllocated(i, allocator_);
        output_names_[i] = nm.get();
    }
}


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

static inline float seg_len2(const cv::Point2f &a, const cv::Point2f &b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
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

static std::vector<std::string> load_keys(const std::string &path) {
    std::ifstream ifs(path);
    std::vector<std::string> ks;
    std::string line;
    while (std::getline(ifs, line))
        ks.push_back(line);
    return ks;
}

static int find_space_k_in_keys(const std::vector<std::string> &keys) {
    for (int i = 0; i < (int) keys.size(); ++i) {
        if (keys[i] == " ")
            return i + 1; // 模型索引 = 字典索引+1（blank=0）
    }
    return -1;
}


Recognizer::Recognizer(const std::wstring &rec_model, const std::string &keys_path, bool use_cuda, const Params &p) :
    rec_(rec_model, use_cuda), p_(p), charset_(load_keys(keys_path)) {

    // 记录 space 的模型索引（若 keys 无空格则 -1 并告警）
    space_k_ = find_space_k_in_keys(charset_);
    if (space_k_ < 0) {
        std::cerr << "[WARN] keys 不包含空格' '，英文很难输出空格；"
                     "请使用包含空格的 keys 或英文专用 rec 模型。\n";
    }

    // 从模型读取真实输入形状（NCHW），遇到动态维(<=0)则保留现有参数
    const auto &in_shapes = rec_.input_shapes();
    if (!in_shapes.empty() && in_shapes[0].size() >= 4) {
        int64_t H = in_shapes[0][2];
        int64_t W = in_shapes[0][3];
        if (H > 0)
            p_.imgH = (int) H;
        if (W > 0)
            p_.imgW = (int) W; // 若为动态宽(= -1/0) 则保持 320 上限+右侧pad
    }
}

void Recognizer::normalize_rec(cv::Mat &img) {
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    cv::subtract(img, cv::Scalar(0.5, 0.5, 0.5), img);
    cv::divide(img, cv::Scalar(0.5, 0.5, 0.5), img);
}

static inline double logsumexp_row(const float *row, int C) {
    float m = row[0];
    for (int i = 1; i < C; ++i)
        m = std::max(m, row[i]);
    double s = 0.0;
    for (int i = 0; i < C; ++i)
        s += std::exp(row[i] - m);
    return m + std::log(std::max(s, 1e-12));
}

std::string Recognizer::ctc_greedy(const cv::Mat &logits_TxC, float *avg_conf) const {
    const int T = logits_TxC.rows, C = logits_TxC.cols;
    std::string out;
    out.reserve(T);
    int prev_k = -1;
    double sum_logp = 0.0;
    int used = 0;

    // 空格检测的缓冲
    int space_run = 0; // 连续帧计数（空格或空白但空格概率较高）
    int blank_run = 0;
    auto flush_space = [&]() {
        if (!p_.insert_space) {
            space_run = 0;
            blank_run = 0;
            return;
        }
        int space_k = space_k_;
        // 情况 A：字典里有空格类别，按概率插入
        if (space_k > 0) {
            if (space_run >= p_.min_space_frames) out.push_back(' ');
        } else if (p_.fallback_space_from_blank) {
            // 情况 B：字典无空格。若连续 blank 足够长，且左右两侧均为字母/数字，插入一个空格
            if (blank_run >= p_.min_blank_frames_for_space) {
                if (!out.empty() && std::isalnum((unsigned char)out.back())) {
                    out.push_back(' ');
                }
            }
        }
        blank_run = 0;
        space_run = 0;
    };

    for (int t = 0; t < T; ++t) {
        const float *row = logits_TxC.ptr<float>(t);
        int k = int(std::max_element(row, row + C) - row); // argmax
        double lse = logsumexp_row(row, C);
        double p_top = std::exp(row[k] - lse);
        double p_space = (space_k_ > 0) ? std::exp(row[space_k_] - lse) : 0.0;

        // 空格片段统计：满足(预测为空格) 或 (预测空白但空格概率也不低)
        bool is_space_like = (space_k_ > 0) && ((k == space_k_) || (k == 0 && p_space >= p_.space_prob));
        bool is_blank = (k == 0);
        if (is_space_like) {
            space_run++;
            blank_run = is_blank ? (blank_run + 1) : blank_run;
        } else {
            // 片段结束，必要时插入空格
            flush_space();

            // 正常 CTC：忽略 blank(0)，对非空且与前一帧不同的 k 输出
            if (k != 0 && k != prev_k) {
                int dict_idx = k - 1;
                if (dict_idx >= 0 && dict_idx < (int) charset_.size()) {
                    out += charset_[dict_idx];
                    sum_logp += (row[k] - lse); // log p_k
                    used++;
                }
            }
            blank_run = is_blank ? (blank_run + 1) : 0;
        }
        prev_k = k;
    }
    flush_space();

    if (avg_conf)
        *avg_conf = used ? (float) std::exp(sum_logp / used) : 0.f;
    return out;
}

RecResult Recognizer::infer(const cv::Mat &crop) const {
    // 判定是否需要滑窗：理论所需宽度（未截断）
    float r = std::max(1e-6f, float(crop.cols) / float(crop.rows));
    int needW = std::max(8, (int) std::ceil(p_.imgH * r)); // 以恒定高缩放
    bool need_tiling = (needW > p_.imgW); // 超过单窗上限则滑窗

    auto run_once = [&](const cv::Mat &roi) -> RecResult {
        // 1) 缩放到目标高/限制宽
        float rr = std::max(1e-6f, float(roi.cols) / float(roi.rows));
        int tarW = std::min(p_.imgW, std::max(8, (int) std::ceil(p_.imgH * rr)));
        cv::Mat img;
        cv::resize(roi, img, cv::Size(tarW, p_.imgH), 0, 0, cv::INTER_CUBIC);
        // 2) 归一化 + pad
        normalize_rec(img);
        if (tarW < p_.imgW) {
            cv::copyMakeBorder(img, img, 0, 0, 0, p_.imgW - tarW, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        // 3) HWC->CHW, BGR->RGB（修复：swapRB 应为 true）
        cv::Mat chw;
        cv::dnn::blobFromImage(img, chw, 1.0, cv::Size(), cv::Scalar(), false, /*swapRB=*/true);
        // 4) ORT 前向
        std::vector<int64_t> ishape{1, 3, p_.imgH, p_.imgW};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input = Ort::Value::CreateTensor<float>(mem, (float *) chw.data, (size_t) chw.total(), ishape.data(),
                                                     ishape.size());
        auto in_names = rec_.input_names();
        auto out_names = rec_.output_names();
        std::vector<const char *> in{in_names[0].c_str()};
        std::vector<const char *> out{out_names[0].c_str()};
        auto outputs = rec_.session().Run(Ort::RunOptions{nullptr}, in.data(), &input, 1, out.data(), 1);
        // 5) 统一 TxC
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        auto shp = info.GetShape();
        const float *ptr = outputs[0].GetTensorData<float>();
        cv::Mat logits_TxC;
        if (shp.size() == 3) {
            int A = (int) shp[1], B = (int) shp[2];
            bool layout_NCT = (A > B);
            int T = layout_NCT ? B : A;
            int C = layout_NCT ? A : B;
            if (!layout_NCT) {
                cv::Mat tmp(T, C, CV_32F, const_cast<float *>(ptr));
                logits_TxC = tmp.clone();
            } else {
                logits_TxC.create(T, C, CV_32F);
                for (int t = 0; t < T; ++t) {
                    const float *p_col = ptr + t;
                    float *dst = logits_TxC.ptr<float>(t);
                    for (int c = 0; c < C; ++c)
                        dst[c] = p_col[c * T];
                }
            }
        } else if (shp.size() == 4) {
            int n = (int) shp[0], a = (int) shp[1], b = (int) shp[2], c = (int) shp[3];
            if (n != 1)
                throw std::runtime_error("rec output N!=1 not supported");
            if (a == 1) {
                int T = b, C = c;
                cv::Mat tmp(T, C, CV_32F, const_cast<float *>(ptr));
                logits_TxC = tmp.clone();
            } else {
                int C = a, T = c;
                logits_TxC.create(T, C, CV_32F);
                for (int t = 0; t < T; ++t) {
                    const float *p_col = ptr + t;
                    float *dst = logits_TxC.ptr<float>(t);
                    for (int k = 0; k < C; ++k)
                        dst[k] = p_col[k * T];
                }
            }
        } else {
            throw std::runtime_error("Unexpected rec output rank: " + std::to_string(shp.size()));
        }
        RecResult rrlt;
        rrlt.text = ctc_greedy(logits_TxC, &rrlt.score);
        return rrlt;
    };

    if (!need_tiling) {
        return run_once(crop);
    }

    // --- 长文本滑窗重组 ---
    // 先将整行缩放到 (H=imgH, W=needW)，再做水平窗口切分
    int capW = std::min(needW, std::max(p_.imgW, p_.max_imgW)); // 控制极端长行避免内存爆
    float scale = float(p_.imgH) / float(crop.rows);
    int scaledW = std::min((int) std::ceil(crop.cols * scale), capW);
    cv::Mat scaled;
    cv::resize(crop, scaled, cv::Size(scaledW, p_.imgH), 0, 0, cv::INTER_CUBIC);

    int win = p_.imgW;
    int step = std::max(8, (int) std::round(win * (1.f - std::clamp(p_.tile_overlap, 0.f, 0.5f))));
    std::string merged;
    double sum_score = 0.0;
    int seg_cnt = 0;
    auto merge_text = [&](const std::string &prev, const std::string &curr) -> std::string {
        int Lmax = std::min({p_.merge_overlap_chars, (int) prev.size(), (int) curr.size()});
        for (int L = Lmax; L >= 1; --L) {
            if (prev.compare((int) prev.size() - L, L, curr, 0, L) == 0) {
                return prev + curr.substr(L);
            }
        }
        return prev + curr;
    };

    for (int x = 0; x < scaledW; x += step) {
        int w = std::min(win, scaledW - x);
        cv::Rect roi(x, 0, w, p_.imgH);
        cv::Mat tile = scaled(roi).clone();
        RecResult part = run_once(tile);
        if (merged.empty())
            merged = part.text;
        else
            merged = merge_text(merged, part.text);
        sum_score += std::clamp((double) part.score, 0.0, 1.0);
        seg_cnt++;
        if (x + w >= scaledW)
            break;
    }
    RecResult rlt;
    rlt.text = merged;
    rlt.score = seg_cnt ? (float) (sum_score / seg_cnt) : 0.f;
    return rlt;
}

static inline float l2(const cv::Point2f &a, const cv::Point2f &b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}
static inline std::array<cv::Point2f, 4> order_quad_TL_TR_BR_BL(const std::vector<cv::Point2f> &q) {
    CV_Assert(q.size() == 4);
    auto sum = [](const cv::Point2f &p) { return p.x + p.y; };
    auto diff = [](const cv::Point2f &p) { return p.y - p.x; };
    int tl = 0, tr = 0, br = 0, bl = 0;
    float minSum = FLT_MAX, maxSum = -FLT_MAX, minDiff = FLT_MAX, maxDiff = -FLT_MAX;
    for (int i = 0; i < 4; ++i) {
        float s = sum(q[i]), d = diff(q[i]);
        if (s < minSum) {
            minSum = s;
            tl = i;
        }
        if (s > maxSum) {
            maxSum = s;
            br = i;
        }
        if (d < minDiff) {
            minDiff = d;
            tr = i;
        }
        if (d > maxDiff) {
            maxDiff = d;
            bl = i;
        }
    }
    return {q[tl], q[tr], q[br], q[bl]};
}

static inline float seg_len(const cv::Point2f &a, const cv::Point2f &b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Mat PPOCR::crop_quad_upright(const cv::Mat &img, const std::vector<cv::Point2f> &q_in) {
    auto qv = order_quad_TL_TR_BR_BL(q_in);
    const cv::Point2f &tl = qv[0];
    const cv::Point2f &tr = qv[1];
    const cv::Point2f &br = qv[2];
    const cv::Point2f &bl = qv[3];

    float w = std::max(seg_len(tl, tr), seg_len(bl, br));
    float h = std::max(seg_len(tl, bl), seg_len(tr, br));
    w = std::max(8.0f, w);
    h = std::max(8.0f, h);

    std::vector<cv::Point2f> src{tl, tr, br, bl};
    std::vector<cv::Point2f> dst{{0, 0}, {w - 1, 0}, {w - 1, h - 1}, {0, h - 1}};

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat line;
    cv::warpPerspective(img, line, M, cv::Size((int) w, (int) h), cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    // 行文本要求横向（宽 >= 高）；若竖排，转成横向
    if (line.rows > line.cols) {
        cv::transpose(line, line);
        cv::flip(line, line, 1);
    }
    return line;
}

PPOCR::PPOCR(const std::wstring &det_model, const std::wstring &rec_model, const std::string &keys_path,
             bool use_cuda) : det_(det_model, use_cuda), rec_(rec_model, keys_path, use_cuda) {}

std::vector<std::pair<TextBox, RecResult>> PPOCR::run(const cv::Mat &bgr) const {
    auto boxes = det_.detect(bgr);

    // 简单行序排序：先按 y，再按 x
    std::sort(boxes.begin(), boxes.end(), [](const TextBox &a, const TextBox &b) {
        float ay = 0.5f * (a.quad[0].y + a.quad[3].y);
        float by = 0.5f * (b.quad[0].y + b.quad[3].y);
        if (std::abs(ay - by) < 10.f)
            return a.quad[0].x < b.quad[0].x;
        return ay < by;
    });

    std::vector<std::pair<TextBox, RecResult>> results;
    results.reserve(boxes.size());

    for (auto &tb: boxes) {
        cv::Mat crop = crop_quad_upright(bgr, tb.quad);
        static int dump_id = 0;
        cv::imwrite(cv::format("rec_crop_%03d.png", dump_id++), crop);
        auto rr = rec_.infer(crop);
        results.emplace_back(tb, rr);
    }
    return results;
}
