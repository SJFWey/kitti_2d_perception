#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "core/detection.h"

namespace cv
{
class Mat;
}

class Detector2D
{
public:
    Detector2D(
        const std::string &model_path,
        float score_threshold = 0.5f,
        int input_height = 384,
        int input_width = 1248,
        std::vector<int> class_allowlist = {1, 2, 3});

    std::vector<Detection> infer(const cv::Mat &image);

private:
    Ort::Env env{nullptr};
    Ort::Session session{nullptr};

    float score_threshold;
    int input_height;
    int input_width;

    std::vector<float> input_tensor_values_;
    std::vector<int64_t> input_shape_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char *> input_name_ptrs_;
    std::vector<const char *> output_name_ptrs_;
    std::unordered_set<int> class_allowlist_;

    void preprocess(const cv::Mat &image);
    void init_io_names();
    bool is_class_allowed(int class_id) const;
    void warmup();
};
