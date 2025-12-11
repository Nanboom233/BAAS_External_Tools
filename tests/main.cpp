#include <iostream>
#include "ppocr.h"

#ifdef _WIN32
#include <Windows.h>
static std::wstring utf8_to_w(const std::string &s) {
    if (s.empty())
        return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring ws(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), &ws[0], n);
    return ws;
}
#endif

int main(int argc, char** argv) {
    bool use_cuda = false;

    cv::Mat img = cv::imread(
            R"(C:\Users\Nanboom233\Desktop\Code\blue_archive_auto_script\core\external_tools\external_tools\tests\test.png)"
    );
    if (img.empty()) {
        std::cerr << "read image failed\n";
        return -1;
    }

    std::wstring det_path =
            LR"(C:\Users\Nanboom233\Desktop\Code\blue_archive_auto_script\core\external_tools\external_tools\resources\paddle_models\PP-OCRv5_server_det\inference.onnx)";
    std::wstring rec_path =
            LR"(C:\Users\Nanboom233\Desktop\Code\blue_archive_auto_script\core\external_tools\external_tools\resources\paddle_models\PP-OCRv5_server_rec\inference.onnx)";

    std::string keys =
            R"(C:\Users\Nanboom233\Desktop\Code\blue_archive_auto_script\core\external_tools\external_tools\resources\paddle_models\ppocrv5_dict.txt)";

    PPOCR ocr(det_path, rec_path, keys, use_cuda);
    auto results = ocr.run(img);

    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    for (auto &pr: results) {
        const auto &tb = pr.first;
        const auto &rr = pr.second;
        std::cout << rr.text << "  score=" << rr.score << "\n";
        for (int i = 0; i < 4; ++i)
            cv::line(img, tb.quad[i], tb.quad[(i + 1) % 4], {0, 255, 0}, 2);
    }
    cv::imwrite("vis.jpg", img);
    std::cout << "Saved to vis.jpg, texts=" << results.size() << " (GPU=" << (use_cuda ? "on" : "off") << ")\n";
    return 0;
}
