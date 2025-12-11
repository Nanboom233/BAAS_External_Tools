#include "onnx_session.h"
#include <opencv2/dnn.hpp>

Ort::Env &OrtEnvHolder::Get() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "baas_ocr");
    return env; // 单例：程序全程只创建一次
}

OnnxSession::OnnxSession(const std::wstring &model_path, bool use_cuda, int intra_threads) {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(intra_threads);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // GPU providers: CUDA first, fallback to DirectML, then CPU.
    if (use_cuda) {
        try {
#ifdef ORT_API_MANUAL_INIT
            // some builds require manual provider registration; omitted here
#endif
#if defined(ORT_SESSION_OPTIONS_APPEND_EXECUTION_PROVIDER_CUDA)
            OrtCUDAProviderOptions cuda_opts{}; // defaults
            opt.AppendExecutionProvider_CUDA(cuda_opts);
#elif defined(OrtSessionOptionsAppendExecutionProvider_CUDA)
            // Older symbol name
            OrtCUDAProviderOptions cuda_opts{};
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(opt, &cuda_opts));
#endif
        } catch (...) {
            // ignore and try DML
        }
        try {
#if defined(ORT_SESSION_OPTIONS_APPEND_EXECUTION_PROVIDER_DML)
            OrtDmlApi* dml_api = nullptr; // if needed by certain builds
            opt.AppendExecutionProvider_DML(0);
#elif defined(OrtSessionOptionsAppendExecutionProvider_DML)
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(opt, 0));
#endif
        } catch (...) {
            // ignore, fallback to CPU
        }
    }

    // 创建会话（若未附加任何 EP，则为 CPU）
    session_ = Ort::Session(OrtEnvHolder::Get(), model_path.c_str(), opt);

    // 初始化输入输出的名称和形状信息
    size_t in_cnt = session_.GetInputCount();
    size_t out_cnt = session_.GetOutputCount();

    input_names_.resize(in_cnt);
    output_names_.resize(out_cnt);
    input_shapes_.resize(in_cnt);

    for (size_t i = 0; i < in_cnt; ++i) {
        auto nm = session_.GetInputNameAllocated(i, allocator_);
        input_names_[i] = nm.get();
        Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        auto shp = info.GetShape();
        for (auto &d: shp) {
            if (d == 0) d = -1;
        }
        input_shapes_[i] = std::move(shp);
    }
    for (size_t i = 0; i < out_cnt; ++i) {
        auto nm = session_.GetOutputNameAllocated(i, allocator_);
        output_names_[i] = nm.get();
    }
}

