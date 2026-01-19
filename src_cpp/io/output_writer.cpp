#include "io/output_writer.h"

#include <iomanip>
#include <iostream>
#include <string>

#include "core/detection.h"
#include "core/sort_tracker.h"

namespace output_writer
{
void print_progress(int current, int total)
{
    if (total <= 0)
    {
        return;
    }
    const int bar_width = 30;
    const float progress = static_cast<float>(current) / static_cast<float>(total);
    const int filled = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < filled)
        {
            std::cout << "=";
        }
        else if (i == filled)
        {
            std::cout << ">";
        }
        else
        {
            std::cout << " ";
        }
    }
    std::cout << "] "
              << std::setw(3) << static_cast<int>(progress * 100.0f) << "% ("
              << current << "/" << total << ")"
              << std::flush;

    if (current >= total)
    {
        std::cout << std::endl;
    }
}

void write_csv_header(std::ofstream &csv)
{
    csv << "frame_id,image_name,class_id,score,x1,y1,x2,y2\n";
    csv << std::fixed << std::setprecision(4);
}

void append_detections(std::ofstream &csv,
                       int frame_id,
                       const std::filesystem::path &image_path,
                       const std::vector<Detection> &detections)
{
    const std::string image_name = image_path.filename().string();
    for (const auto &det : detections)
    {
        csv << frame_id << ","
            << image_name << ","
            << det.class_id << ","
            << det.score << ","
            << det.x1 << ","
            << det.y1 << ","
            << det.x2 << ","
            << det.y2 << "\n";
    }
}

void write_track_csv_header(std::ofstream &csv)
{
    csv << "frame_id,image_name,track_id,class_id,score,x1,y1,x2,y2\n";
    csv << std::fixed << std::setprecision(4);
}

void append_tracks(std::ofstream &csv,
                   int frame_id,
                   const std::filesystem::path &image_path,
                   const std::vector<TrackedObject> &tracks)
{
    const std::string image_name = image_path.filename().string();
    for (const auto &track : tracks)
    {
        csv << frame_id << ","
            << image_name << ","
            << track.track_id << ","
            << track.class_id << ","
            << track.score << ","
            << track.x1 << ","
            << track.y1 << ","
            << track.x2 << ","
            << track.y2 << "\n";
    }
}
} // namespace output_writer
