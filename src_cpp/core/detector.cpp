#include "core/detector.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <opencv2/dnn/dnn.hpp>

Detector2D::Detector2D(
    const std::string &model_path,
    float score_threshold,
    int input_height,
    int input_width,
    std::vector<int> class_allowlist)
    : env(ORT_LOGGING_LEVEL_WARNING, "Detector2D")
    , score_threshold(score_threshold)
    , input_height(input_height)
    , input_width(input_width)
{
    if (input_height <= 0 || input_width <= 0)
    {
        throw std::invalid_argument("Detector2D input dimensions must be positive.");
    }
    if (!class_allowlist.empty())
    {
        class_allowlist_.insert(class_allowlist.begin(), class_allowlist.end());
    }
    input_shape_ = {1, 3, input_height, input_width};
    const size_t input_size = static_cast<size_t>(1) * 3 * input_height * input_width;
    input_tensor_values_.assign(input_size, 0.0f);

    Ort::SessionOptions session_options;
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    session_options.SetIntraOpNumThreads(static_cast<int>(hw_threads));
    session_options.SetInterOpNumThreads(1);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    try
    {
        session = Ort::Session(env, model_path.c_str(), session_options);
        std::cout << ">>> ORT threads: intra=" << hw_threads
                  << ", inter=1, execution=PARALLEL" << std::endl;
        init_io_names();
        warmup();
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "Failed to initialize detector: " << e.what() << std::endl;
        throw;
    }
}

void Detector2D::init_io_names()
{
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = session.GetInputCount();
    const size_t output_count = session.GetOutputCount();
    if (input_count != 1)
    {
        throw std::runtime_error("Detector2D expects exactly 1 input tensor.");
    }
    if (output_count < 3)
    {
        throw std::runtime_error("Detector2D expects at least 3 output tensors.");
    }

    input_names_.clear();
    output_names_.clear();
    input_names_.reserve(input_count);
    output_names_.reserve(3);

    for (size_t i = 0; i < input_count; ++i)
    {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_names_.push_back(name.get());
    }
    for (size_t i = 0; i < 3; ++i)
    {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names_.push_back(name.get());
    }

    input_name_ptrs_.clear();
    output_name_ptrs_.clear();
    for (const auto &name : input_names_)
    {
        input_name_ptrs_.push_back(name.c_str());
    }
    for (const auto &name : output_names_)
    {
        output_name_ptrs_.push_back(name.c_str());
    }
}

bool Detector2D::is_class_allowed(int class_id) const
{
    if (class_allowlist_.empty())
    {
        return true;
    }
    return class_allowlist_.find(class_id) != class_allowlist_.end();
}

void Detector2D::warmup()
{
    std::fill(input_tensor_values_.begin(), input_tensor_values_.end(), 0.0f);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_shape_.data(),
        input_shape_.size());

    session.Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs_.data(), &input_tensor, input_name_ptrs_.size(),
        output_name_ptrs_.data(), output_name_ptrs_.size());
}

void Detector2D::preprocess(const cv::Mat &image)
{
    const size_t expected = static_cast<size_t>(1) * 3 * input_height * input_width;
    if (input_tensor_values_.size() != expected)
    {
        input_tensor_values_.assign(expected, 0.0f);
        input_shape_ = {1, 3, input_height, input_width};
    }

    const int sizes[] = {1, 3, input_height, input_width};
    cv::Mat blob(4, sizes, CV_32F, input_tensor_values_.data());
    cv::dnn::blobFromImage(
        image,
        blob,
        1.0f / 255.0f,
        cv::Size(input_width, input_height),
        cv::Scalar(),
        true,
        false,
        CV_32F);
}

std::vector<Detection> Detector2D::infer(const cv::Mat &image)
{
    preprocess(image);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_shape_.data(),
        input_shape_.size());

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs_.data(), &input_tensor, input_name_ptrs_.size(),
        output_name_ptrs_.data(), output_name_ptrs_.size());

    if (output_tensors.size() < 3)
    {
        throw std::runtime_error("Detector2D expected 3 outputs but received fewer.");
    }

    float *float_boxes = output_tensors[0].GetTensorMutableData<float>();
    int64_t *int_labels = output_tensors[1].GetTensorMutableData<int64_t>();
    float *float_scores = output_tensors[2].GetTensorMutableData<float>();

    size_t num_detections = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[0];

    std::vector<Detection> results;
    float x_scale = static_cast<float>(image.cols) / static_cast<float>(input_width);
    float y_scale = static_cast<float>(image.rows) / static_cast<float>(input_height);

    for (size_t i = 0; i < num_detections; ++i)
    {
        float score = float_scores[i];
        if (score < score_threshold)
            continue;

        int class_id = static_cast<int>(int_labels[i]);
        if (!is_class_allowed(class_id))
            continue;

        Detection det;
        det.score = score;
        det.class_id = class_id;
        det.x1 = float_boxes[i * 4 + 0] * x_scale;
        det.y1 = float_boxes[i * 4 + 1] * y_scale;
        det.x2 = float_boxes[i * 4 + 2] * x_scale;
        det.y2 = float_boxes[i * 4 + 3] * y_scale;

        results.push_back(det);
    }

    return results;
}
