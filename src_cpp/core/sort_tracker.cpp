#include "core/sort_tracker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include "core/hungarian.h"

namespace
{
constexpr std::array<float, 7> kBaseProcessNoise = {
    1e-2f, 1e-2f, 1e-1f, 1e-3f, 5e-2f, 5e-2f, 1e-1f};
constexpr std::array<float, 4> kBaseMeasurementNoise = {
    1e-1f, 1e-1f, 1e-1f, 1e-2f};
constexpr float kMinNoiseScale = 1e-6f;

struct NoiseScales
{
    float process = 1.0f;
    float measurement = 1.0f;
};

NoiseScales noise_scales_for_class(int class_id)
{
    if (class_id == 2 || class_id == 3)
    {
        return {2.0f, 1.2f};
    }
    return {1.0f, 1.0f};
}
cv::Rect2f clamp_rect(const cv::Rect2f &rect)
{
    float w = std::max(rect.width, 1.0f);
    float h = std::max(rect.height, 1.0f);
    return cv::Rect2f(rect.x, rect.y, w, h);
}

cv::Rect2f detection_to_rect(const Detection &det)
{
    return clamp_rect(cv::Rect2f(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1));
}

bool rects_overlap(const cv::Rect2f &a, const cv::Rect2f &b)
{
    return (a.x < b.x + b.width) && (b.x < a.x + a.width) &&
           (a.y < b.y + b.height) && (b.y < a.y + a.height);
}

float rect_iou(const cv::Rect2f &a, const cv::Rect2f &b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;
    float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    float union_area = area_a + area_b - inter_area;
    if (union_area <= 0.0f)
    {
        return 0.0f;
    }
    return inter_area / (union_area + 1e-6f);
}

struct AssignmentResult
{
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

AssignmentResult assign_detections_to_tracks(const std::vector<cv::Rect2f> &track_boxes,
                                            const std::vector<int> &track_classes,
                                            const std::vector<cv::Rect2f> &det_boxes,
                                            const std::vector<int> &det_classes,
                                            float iou_threshold,
                                            bool match_class)
{
    AssignmentResult result;
    const int num_tracks = static_cast<int>(track_boxes.size());
    const int num_dets = static_cast<int>(det_boxes.size());
    if (num_tracks == 0)
    {
        result.unmatched_detections.resize(num_dets);
        std::iota(result.unmatched_detections.begin(), result.unmatched_detections.end(), 0);
        return result;
    }
    if (num_dets == 0)
    {
        result.unmatched_tracks.resize(num_tracks);
        std::iota(result.unmatched_tracks.begin(), result.unmatched_tracks.end(), 0);
        return result;
    }

    constexpr float kClassMismatchCost = 1e6f;
    std::vector<std::vector<float>> iou_matrix(num_tracks, std::vector<float>(num_dets, 0.0f));
    std::vector<std::vector<float>> cost_matrix(num_tracks, std::vector<float>(num_dets, 1.0f));
    for (int i = 0; i < num_tracks; ++i)
    {
        for (int j = 0; j < num_dets; ++j)
        {
            if (match_class && track_classes[i] != det_classes[j])
            {
                cost_matrix[i][j] = kClassMismatchCost;
                continue;
            }
            if (!rects_overlap(track_boxes[i], det_boxes[j]))
            {
                continue;
            }
            float iou = rect_iou(track_boxes[i], det_boxes[j]);
            iou_matrix[i][j] = iou;
            cost_matrix[i][j] = 1.0f - iou;
        }
    }

    std::vector<int> assignment = hungarian::solve(cost_matrix);
    std::vector<char> det_used(num_dets, false);
    for (int i = 0; i < num_tracks; ++i)
    {
        int j = (i < static_cast<int>(assignment.size())) ? assignment[i] : -1;
        if (j >= 0 && j < num_dets)
        {
            if (match_class && track_classes[i] != det_classes[j])
            {
                result.unmatched_tracks.push_back(i);
                continue;
            }
            float iou = iou_matrix[i][j];
            if (iou >= iou_threshold)
            {
                result.matches.emplace_back(i, j);
                det_used[j] = true;
            }
            else
            {
                result.unmatched_tracks.push_back(i);
            }
        }
        else
        {
            result.unmatched_tracks.push_back(i);
        }
    }

    for (int j = 0; j < num_dets; ++j)
    {
        if (!det_used[j])
        {
            result.unmatched_detections.push_back(j);
        }
    }

    return result;
}

cv::Mat measurement_from_rect(const cv::Rect2f &bbox)
{
    cv::Rect2f rect = clamp_rect(bbox);
    float w = rect.width;
    float h = rect.height;
    float cx = rect.x + w / 2.0f;
    float cy = rect.y + h / 2.0f;
    float s = w * h;
    float r = w / h;
    cv::Mat measurement(4, 1, CV_32F);
    measurement.at<float>(0) = cx;
    measurement.at<float>(1) = cy;
    measurement.at<float>(2) = s;
    measurement.at<float>(3) = r;
    return measurement;
}

cv::Rect2f rect_from_state(const cv::Mat &state)
{
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float s = std::max(state.at<float>(2), 1.0f);
    float r = std::max(state.at<float>(3), 0.1f);
    float w = std::sqrt(s * r);
    float h = s / w;
    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    return cv::Rect2f(x1, y1, w, h);
}
} // namespace

TrackedObject::TrackedObject(int track_id_,
                             int class_id_,
                             float score_,
                             const cv::Rect2f &bbox,
                             int age_,
                             int hits_,
                             int time_since_update_)
    : track_id(track_id_),
      class_id(class_id_),
      score(score_),
      x1(bbox.x),
      y1(bbox.y),
      x2(bbox.x + bbox.width),
      y2(bbox.y + bbox.height),
      age(age_),
      hits(hits_),
      time_since_update(time_since_update_)
{
}

SortTracker::SortKalmanFilter::SortKalmanFilter()
{
    kf_ = cv::KalmanFilter(7, 4, 0, CV_32F);
    kf_.transitionMatrix = (cv::Mat_<float>(7, 7) << 1, 0, 0, 0, 1, 0, 0,
                              0, 1, 0, 0, 0, 1, 0,
                              0, 0, 1, 0, 0, 0, 1,
                              0, 0, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0, 1, 0,
                              0, 0, 0, 0, 0, 0, 1);
    kf_.measurementMatrix = (cv::Mat_<float>(4, 7) << 1, 0, 0, 0, 0, 0, 0,
                               0, 1, 0, 0, 0, 0, 0,
                               0, 0, 1, 0, 0, 0, 0,
                               0, 0, 0, 1, 0, 0, 0);

    set_noise_scales(1.0f, 1.0f);

    kf_.errorCovPost = cv::Mat::eye(7, 7, CV_32F);
}

void SortTracker::SortKalmanFilter::init(const cv::Rect2f &bbox)
{
    cv::Mat measurement = measurement_from_rect(bbox);
    kf_.statePost = cv::Mat::zeros(7, 1, CV_32F);
    kf_.statePost.at<float>(0) = measurement.at<float>(0);
    kf_.statePost.at<float>(1) = measurement.at<float>(1);
    kf_.statePost.at<float>(2) = measurement.at<float>(2);
    kf_.statePost.at<float>(3) = measurement.at<float>(3);
    kf_.statePost.at<float>(4) = 0.0f;
    kf_.statePost.at<float>(5) = 0.0f;
    kf_.statePost.at<float>(6) = 0.0f;
    kf_.statePre = kf_.statePost.clone();
    initialized_ = true;
}

cv::Rect2f SortTracker::SortKalmanFilter::predict()
{
    if (!initialized_)
    {
        return cv::Rect2f();
    }
    cv::Mat prediction = kf_.predict();
    // kf_.statePost = prediction.clone();
    return rect_from_state(prediction);
}

void SortTracker::SortKalmanFilter::update(const cv::Rect2f &bbox)
{
    if (!initialized_)
    {
        init(bbox);
        return;
    }
    cv::Mat measurement = measurement_from_rect(bbox);
    kf_.correct(measurement);
}

void SortTracker::SortKalmanFilter::coast()
{
    if (!initialized_)
    {
        return;
    }
    kf_.statePost = kf_.statePre.clone();
}

cv::Rect2f SortTracker::SortKalmanFilter::current_bbox() const
{
    if (!initialized_)
    {
        return cv::Rect2f();
    }
    return rect_from_state(kf_.statePost);
}

bool SortTracker::SortKalmanFilter::initialized() const
{
    return initialized_;
}

void SortTracker::SortKalmanFilter::set_noise_scales(float process_scale, float measurement_scale)
{
    const float process = std::max(process_scale, kMinNoiseScale);
    const float measurement = std::max(measurement_scale, kMinNoiseScale);

    kf_.processNoiseCov = cv::Mat::zeros(7, 7, CV_32F);
    for (int i = 0; i < static_cast<int>(kBaseProcessNoise.size()); ++i)
    {
        kf_.processNoiseCov.at<float>(i, i) = kBaseProcessNoise[i] * process;
    }

    kf_.measurementNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
    for (int i = 0; i < static_cast<int>(kBaseMeasurementNoise.size()); ++i)
    {
        kf_.measurementNoiseCov.at<float>(i, i) = kBaseMeasurementNoise[i] * measurement;
    }
}

SortTracker::Track::Track(int track_id_, const Detection &det, const cv::Rect2f &bbox_)
    : track_id(track_id_),
      class_id(det.class_id),
      score(det.score),
      age(1),
      hits(1),
      time_since_update(0),
      bbox(bbox_)
{
    const NoiseScales scales = noise_scales_for_class(det.class_id);
    kf.set_noise_scales(scales.process, scales.measurement);
    kf.init(bbox_);
}

SortTracker::SortTracker(int max_age, int min_hits, float iou_threshold, bool match_class, int output_max_age)
    : max_age_(max_age),
      min_hits_(min_hits),
      iou_threshold_(iou_threshold),
      match_class_(match_class),
      output_max_age_(std::max(0, output_max_age)),
      next_track_id_(1),
      frame_index_(0)
{
}

std::vector<TrackedObject> SortTracker::update(const std::vector<Detection> &detections)
{
    frame_index_++;
    std::vector<cv::Rect2f> det_boxes;
    std::vector<int> det_classes;
    det_boxes.reserve(detections.size());
    det_classes.reserve(detections.size());
    for (const auto &det : detections)
    {
        det_boxes.push_back(detection_to_rect(det));
        det_classes.push_back(det.class_id);
    }

    std::vector<cv::Rect2f> track_boxes;
    std::vector<int> track_classes;
    track_boxes.reserve(tracks_.size());
    track_classes.reserve(tracks_.size());
    for (auto &track : tracks_)
    {
        track.bbox = track.kf.predict();
        track.age += 1;
        track.time_since_update += 1;
        track_boxes.push_back(track.bbox);
        track_classes.push_back(track.class_id);
    }

    AssignmentResult assignment = assign_detections_to_tracks(
        track_boxes,
        track_classes,
        det_boxes,
        det_classes,
        iou_threshold_,
        match_class_);

    for (const auto &match : assignment.matches)
    {
        int track_idx = match.first;
        int det_idx = match.second;
        Track &track = tracks_[track_idx];
        const Detection &det = detections[det_idx];
        track.kf.update(det_boxes[det_idx]);
        track.bbox = track.kf.current_bbox();
        track.time_since_update = 0;
        track.hits += 1;
        track.score = det.score;
        track.class_id = det.class_id;
    }

    for (int track_idx : assignment.unmatched_tracks)
    {
        Track &track = tracks_[track_idx];
        track.kf.coast();
        track.bbox = track.kf.current_bbox();
    }

    for (int det_idx : assignment.unmatched_detections)
    {
        const Detection &det = detections[det_idx];
        tracks_.emplace_back(next_track_id_++, det, det_boxes[det_idx]);
        tracks_.back().bbox = tracks_.back().kf.current_bbox();
    }

    std::vector<Track> active_tracks;
    active_tracks.reserve(tracks_.size());
    for (const auto &track : tracks_)
    {
        if (track.time_since_update <= max_age_)
        {
            active_tracks.push_back(track);
        }
    }
    tracks_.swap(active_tracks);

    std::vector<TrackedObject> outputs;
    for (const auto &track : tracks_)
    {
        if (track.hits < min_hits_ || track.time_since_update > output_max_age_)
        {
            continue;
        }

        TrackedObject obj(track.track_id,
                          track.class_id,
                          track.score,
                          track.bbox,
                          track.age,
                          track.hits,
                          track.time_since_update);
        outputs.push_back(obj);
    }

    return outputs;
}
