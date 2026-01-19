#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "core/detection.h"

struct TrackedObject
{
    int track_id = -1;
    int class_id = -1;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    int age = 0;
    int hits = 0;
    int time_since_update = 0;

    TrackedObject() = default;
    TrackedObject(int track_id_,
                  int class_id_,
                  float score_,
                  const cv::Rect2f &bbox,
                  int age_,
                  int hits_,
                  int time_since_update_);
};

class SortTracker
{
public:
    SortTracker(int max_age = 3,
                int min_hits = 3,
                float iou_threshold = 0.3f,
                bool match_class = true,
                int output_max_age = 1);

    std::vector<TrackedObject> update(const std::vector<Detection> &detections);

private:
    class SortKalmanFilter
    {
    public:
        SortKalmanFilter();

        void init(const cv::Rect2f &bbox);
        cv::Rect2f predict();
        void update(const cv::Rect2f &bbox);
        void coast();
        cv::Rect2f current_bbox() const;
        bool initialized() const;
        void set_noise_scales(float process_scale, float measurement_scale);

    private:
        cv::KalmanFilter kf_;
        bool initialized_ = false;
    };

    struct Track
    {
        int track_id = -1;
        int class_id = -1;
        float score = 0.0f;
        int age = 0;
        int hits = 0;
        int time_since_update = 0;
        cv::Rect2f bbox;
        SortKalmanFilter kf;

        Track() = default;
        Track(int track_id_, const Detection &det, const cv::Rect2f &bbox_);
    };

    int max_age_;
    int min_hits_;
    float iou_threshold_;
    bool match_class_;
    int output_max_age_;
    int next_track_id_ = 1;
    int frame_index_ = 0;
    std::vector<Track> tracks_;
};
