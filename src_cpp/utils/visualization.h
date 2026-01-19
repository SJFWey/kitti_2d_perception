#pragma once

#include <vector>

#include <opencv2/core.hpp>

struct Detection;
struct TrackedObject;

void draw_objects(cv::Mat &img, const std::vector<Detection> &objects);
cv::Scalar track_color(int track_id);
void draw_track_tails(cv::Mat &img,
                      const std::vector<TrackedObject> &tracks,
                      int max_tail_length = 30,
                      int max_idle_frames = 30);
void draw_tracked_objects(cv::Mat &img, const std::vector<TrackedObject> &tracks);
