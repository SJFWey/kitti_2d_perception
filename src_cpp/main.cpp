#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <opencv2/opencv.hpp>
#include "detector.h"

void draw_objects(cv::Mat &img, const std::vector<Detection> &objects)
{
    std::map<int, std::string> class_names = {
        {1, "Car"},
        {2, "Pedestrian"},
        {3, "Cyclist"}};

    std::map<int, cv::Scalar> colors = {
        {1, cv::Scalar(0, 255, 0)},
        {2, cv::Scalar(0, 0, 255)},
        {3, cv::Scalar(255, 0, 0)}};

    for (const auto &obj : objects)
    {
        cv::Scalar color = colors.count(obj.class_id) ? colors[obj.class_id] : cv::Scalar(255, 255, 255);
        std::string name = class_names.count(obj.class_id) ? class_names[obj.class_id] : "Unknown";

        std::string label = name + ": " + std::to_string(obj.score).substr(0, 4);

        cv::Rect rect(
            cv::Point(static_cast<int>(obj.x1), static_cast<int>(obj.y1)),
            cv::Point(static_cast<int>(obj.x2), static_cast<int>(obj.y2)));
        cv::rectangle(img, rect, color, 2);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(rect.y, labelSize.height);

        cv::rectangle(img,
                      cv::Point(rect.x, top - labelSize.height),
                      cv::Point(rect.x + labelSize.width, top + baseLine),
                      color, cv::FILLED);

        cv::putText(img, label, cv::Point(rect.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char **argv)
{
    std::string img_path = "../data/kitti_detection/images/000796.png";
    if (argc > 1)
        img_path = argv[1];

    std::string model_path = "../models/model_v2.onnx";

    std::cout << ">>> Init Detector..." << std::endl;
    try
    {
        Detector2D detector(model_path);

        std::cout << ">>> Loading image: " << img_path << std::endl;
        cv::Mat img = cv::imread(img_path);

        if (img.empty())
        {
            std::cerr << "!!! Error: Could not load image. Check path!" << std::endl;
            return -1;
        }

        std::cout << ">>> Running Inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto results = detector.infer(img);

        std::vector<Detection> filtered;
        filtered.reserve(results.size());
        for (const auto &det : results)
        {
            if (det.class_id == 1 || det.class_id == 2 || det.class_id == 3)
                filtered.push_back(det);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << ">>> Done. Found " << filtered.size() << " objects." << std::endl;
        std::cout << ">>> Inference Time: " << duration.count() << " ms" << std::endl;

        draw_objects(img, filtered);

        std::string output_path = "../output/result.jpg";
        cv::imwrite(output_path, img);
        std::cout << ">>> Result saved to: build/" << output_path << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}