#include "utils/visualization.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/detection.h"
#include "core/sort_tracker.h"

namespace
{
const std::map<int, std::string> kClassNames = {
    {1, "Car"},
    {2, "Pedestrian"},
    {3, "Cyclist"}};

const std::map<int, cv::Scalar> kClassColors = {
    {1, cv::Scalar(0, 255, 0)},
    {2, cv::Scalar(0, 0, 255)},
    {3, cv::Scalar(255, 0, 0)}};

const std::string kUnknownClass = "Unknown";

const std::string &class_name(int class_id)
{
    auto it = kClassNames.find(class_id);
    return it != kClassNames.end() ? it->second : kUnknownClass;
}

cv::Scalar class_color(int class_id)
{
    auto it = kClassColors.find(class_id);
    if (it != kClassColors.end())
    {
        return it->second;
    }
    return cv::Scalar(255, 255, 255);
}

void draw_labeled_box(cv::Mat &img,
                      const cv::Rect &rect,
                      const std::string &label,
                      const cv::Scalar &color)
{
    cv::rectangle(img, rect, color, 2);

    int base_line = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
    int top = std::max(rect.y, label_size.height);

    cv::rectangle(img,
                  cv::Point(rect.x, top - label_size.height),
                  cv::Point(rect.x + label_size.width, top + base_line),
                  color, cv::FILLED);

    cv::putText(img, label, cv::Point(rect.x, top),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}
} // namespace

void draw_objects(cv::Mat &img, const std::vector<Detection> &objects)
{
    for (const auto &obj : objects)
    {
        const cv::Scalar color = class_color(obj.class_id);
        const std::string &name = class_name(obj.class_id);

        std::ostringstream label_stream;
        label_stream << name << ": " << std::fixed << std::setprecision(2) << obj.score;
        std::string label = label_stream.str();

        cv::Rect rect(
            cv::Point(static_cast<int>(obj.x1), static_cast<int>(obj.y1)),
            cv::Point(static_cast<int>(obj.x2), static_cast<int>(obj.y2)));
        draw_labeled_box(img, rect, label, color);
    }
}

cv::Scalar track_color(int track_id)
{
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(180, 180, 0),
        cv::Scalar(180, 0, 180),
        cv::Scalar(0, 180, 180),
        cv::Scalar(128, 128, 128)};
    if (palette.empty())
    {
        return cv::Scalar(255, 255, 255);
    }
    int idx = std::abs(track_id) % static_cast<int>(palette.size());
    return palette[idx];
}

void draw_track_tails(cv::Mat &img,
                      const std::vector<TrackedObject> &tracks,
                      int max_tail_length,
                      int max_idle_frames)
{
    struct TrackTrail
    {
        std::deque<cv::Point2f> points;
        int last_seen = 0;
    };

    struct TrailState
    {
        std::map<int, TrackTrail> trails;
        int frame_index = 0;
    };

    static TrailState state;
    state.frame_index += 1;
    const int frame_index = state.frame_index;
    const int tail_length = std::max(1, max_tail_length);
    const int max_idle = std::max(tail_length, max_idle_frames);

    for (const auto &obj : tracks)
    {
        cv::Point2f center((obj.x1 + obj.x2) * 0.5f, (obj.y1 + obj.y2) * 0.5f);
        auto &trail = state.trails[obj.track_id];
        trail.points.push_back(center);
        while (static_cast<int>(trail.points.size()) > tail_length)
        {
            trail.points.pop_front();
        }
        trail.last_seen = frame_index;
    }

    for (auto it = state.trails.begin(); it != state.trails.end();)
    {
        if (frame_index - it->second.last_seen > max_idle)
        {
            it = state.trails.erase(it);
        }
        else
        {
            ++it;
        }
    }

    for (const auto &obj : tracks)
    {
        auto it = state.trails.find(obj.track_id);
        if (it == state.trails.end())
        {
            continue;
        }
        const auto &points = it->second.points;
        if (points.size() < 2)
        {
            continue;
        }
        cv::Scalar base = track_color(obj.track_id);
        const int point_count = static_cast<int>(points.size());
        for (int i = 1; i < point_count; ++i)
        {
            float t = static_cast<float>(i) / static_cast<float>(point_count - 1);
            float fade = 0.3f + 0.7f * t;
            cv::Scalar color(base[0] * fade, base[1] * fade, base[2] * fade);
            int thickness = std::max(1, static_cast<int>(std::round(1.0f + 2.0f * t)));
            cv::line(img, points[i - 1], points[i], color, thickness, cv::LINE_AA);
        }
    }
}

void draw_tracked_objects(cv::Mat &img, const std::vector<TrackedObject> &tracks)
{
    for (const auto &obj : tracks)
    {
        const cv::Scalar color = track_color(obj.track_id);
        const std::string &name = class_name(obj.class_id);

        std::ostringstream label_stream;
        label_stream << "ID " << obj.track_id << " " << name << ": "
                     << std::fixed << std::setprecision(2) << obj.score;
        std::string label = label_stream.str();

        cv::Rect rect(
            cv::Point(static_cast<int>(obj.x1), static_cast<int>(obj.y1)),
            cv::Point(static_cast<int>(obj.x2), static_cast<int>(obj.y2)));
        draw_labeled_box(img, rect, label, color);
    }
}
