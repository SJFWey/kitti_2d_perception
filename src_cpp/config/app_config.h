#pragma once

#include <string>
#include <vector>

#include "config/ini_config.h"

namespace app_config
{
inline constexpr float kUnsetFloat = -1.0f;
inline constexpr int kUnsetInt = -1;
inline constexpr int kUnsetMaxFrames = -2;

struct AppOptions
{
    std::string input_path;
    std::string sequence_id;
    std::string model_path;
    std::string output_dir;
    float score_threshold = kUnsetFloat;
    int input_height = kUnsetInt;
    int input_width = kUnsetInt;
    std::vector<int> class_ids;
    bool save_vis = true;
    int max_frames = kUnsetMaxFrames;
    int track_max_age = kUnsetInt;
    int track_output_max_age = kUnsetInt;
    int track_min_hits = kUnsetInt;
    float track_iou_threshold = kUnsetFloat;
};

struct CliOverrides
{
    bool input_path = false;
    bool sequence_id = false;
    bool model_path = false;
    bool output_dir = false;
    bool score_threshold = false;
    bool max_frames = false;
    bool no_save_vis = false;
};

bool apply_config(const ini::IniConfig &config,
                  const CliOverrides &cli,
                  AppOptions &options);

std::vector<std::string> validate_options(const AppOptions &options);
} // namespace app_config
