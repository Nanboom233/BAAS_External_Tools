// filepath: c:\Users\Nanboom233\Desktop\Code\blue_archive_auto_script\core\external_tools\external_tools\include\onnx_session.h
#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

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

