#include "detector.h"
#include <iostream>

Detector2D::Detector2D(
    const std::string &model_path,
    float score_threshold,
    int input_height,
    int input_width)
{
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Detector2D");

    this->score_threshold = score_threshold;
    this->input_height = input_height;
    this->input_width = input_width;

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    try
    {
        session = Ort::Session(env, model_path.c_str(), session_options);
        warmup();
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "Failed to initialize detector: " << e.what() << std::endl;
        throw;
    }
}

void Detector2D::warmup()
{
    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width, 0.0f);
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};

    const char *input_names[] = {"input"};
    const char *output_names[] = {"boxes", "labels", "scores"};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 3);
}

void Detector2D::preprocess(const cv::Mat &image, std::vector<float> &input_tensor_values, std::vector<int64_t> &input_shape)
{
    cv::Mat rgb_img;
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb_img, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    input_shape = {1, 3, input_height, input_width};

    input_tensor_values.resize(1 * 3 * input_height * input_width);

    float *r_ptr = input_tensor_values.data();
    float *g_ptr = r_ptr + input_height * input_width;
    float *b_ptr = g_ptr + input_height * input_width;

    for (int i = 0; i < input_height * input_width; ++i)
    {
        unsigned char r = resized.data[i * 3 + 0];
        unsigned char g = resized.data[i * 3 + 1];
        unsigned char b = resized.data[i * 3 + 2];

        r_ptr[i] = r / 255.0f;
        g_ptr[i] = g / 255.0f;
        b_ptr[i] = b / 255.0f;
    }
}

std::vector<Detection> Detector2D::infer(const cv::Mat &image)
{
    std::vector<float> input_tensor_values;
    std::vector<int64_t> input_shape;
    preprocess(image, input_tensor_values, input_shape);

    const char *input_names[] = {"input"};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    const char *output_names[] = {"boxes", "labels", "scores"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 3);

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
        if (class_id != 1 && class_id != 2 && class_id != 3)
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