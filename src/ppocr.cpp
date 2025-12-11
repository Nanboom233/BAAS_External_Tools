#include <opencv2/opencv.hpp>
#include <ppocr.h>

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