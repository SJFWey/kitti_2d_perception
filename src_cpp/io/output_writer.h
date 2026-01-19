#pragma once

#include <filesystem>
#include <fstream>
#include <vector>

struct Detection;
struct TrackedObject;

namespace output_writer
{
void print_progress(int current, int total);
void write_csv_header(std::ofstream &csv);
void append_detections(std::ofstream &csv,
                       int frame_id,
                       const std::filesystem::path &image_path,
                       const std::vector<Detection> &detections);
void write_track_csv_header(std::ofstream &csv);
void append_tracks(std::ofstream &csv,
                   int frame_id,
                   const std::filesystem::path &image_path,
                   const std::vector<TrackedObject> &tracks);
} // namespace output_writer
