#include <algorithm>
#include <cctype>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <recognizer.h>

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