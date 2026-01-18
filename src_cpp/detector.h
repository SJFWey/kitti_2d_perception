#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

class Detector2D
{
public:
    Detector2D(
        const std::string &model_path,
        float score_threshold = 0.4f,
        int input_height = 384,
        int input_width = 1248);

    std::vector<Detection> infer(const cv::Mat &image);

private:
    Ort::Env env{nullptr};
    Ort::Session session{nullptr};

    float score_threshold = 0.5f;
    int input_height = 384;
    int input_width = 1248;

    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std = {0.229f, 0.224f, 0.225f};

    void preprocess(
        const cv::Mat &image,
        std::vector<float> &input_tensor_values,
        std::vector<int64_t> &input_shape);
    void warmup();
};