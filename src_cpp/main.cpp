#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#if __has_include(<CLI/CLI.hpp>)
#include <CLI/CLI.hpp>
#else
#include "CLI11.hpp"
#endif
#include "config/app_config.h"
#include "core/detector.h"
#include "io/input_resolver.h"
#include "core/sort_tracker.h"
#include "utils/visualization.h"
#include "io/output_writer.h"

namespace fs = std::filesystem;

namespace
{
const fs::path kDefaultPublicConfig = fs::path("configs") / "public" / "default.ini";
} // namespace

using Clock = std::chrono::high_resolution_clock;

static long long ms_between(const Clock::time_point &start, const Clock::time_point &end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char **argv)
{
    app_config::AppOptions options;
    std::string config_path;

    CLI::App app{"Perception2D"};
    auto *config_opt = app.add_option("--config", config_path, "Path to INI config file");
    auto *input_opt = app.add_option("-i,--input", options.input_path, "Input image file or directory");
    auto *sequence_opt = app.add_option("-s,--sequence", options.sequence_id, "Sequence id under input root (required)");
    auto *model_path_opt = app.add_option("--model-path", options.model_path, "Path to ONNX model");
    auto *output_dir_opt = app.add_option("-o,--output", options.output_dir, "Output root directory");
    auto *score_opt = app.add_option("--score-threshold", options.score_threshold, "Detection score threshold");
    auto *max_frames_opt = app.add_option("--max-frames", options.max_frames, "Max frames to process (-1 for all)");

    bool no_save_vis = false;
    app.add_flag("--no-save-vis", no_save_vis, "Disable visualization output");

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e)
    {
        if (e.is_help())
        {
            std::cout << app.help() << std::endl;
            return e.get_exit_code();
        }
        std::cerr << "CLI error: " << e.what() << "\n"
                  << app.help() << std::endl;
        return e.get_exit_code();
    }

    const bool input_from_cli = input_opt->count() > 0;
    const bool sequence_from_cli = sequence_opt->count() > 0;
    const bool model_from_cli = model_path_opt->count() > 0;
    const bool output_from_cli = output_dir_opt->count() > 0;
    const bool score_from_cli = score_opt->count() > 0;
    const bool max_frames_from_cli = max_frames_opt->count() > 0;

    app_config::CliOverrides cli;
    cli.input_path = input_from_cli;
    cli.sequence_id = sequence_from_cli;
    cli.model_path = model_from_cli;
    cli.output_dir = output_from_cli;
    cli.score_threshold = score_from_cli;
    cli.max_frames = max_frames_from_cli;
    cli.no_save_vis = no_save_vis;

    if (!fs::exists(kDefaultPublicConfig))
    {
        std::cerr << "Default config not found: " << kDefaultPublicConfig << "\n"
                  << "Please run from project root directory.\n";
        return 1;
    }
    const bool config_required = config_opt->count() > 0;
    std::vector<ini::IniPath> config_paths = {
        {kDefaultPublicConfig, true},
    };
    if (!config_path.empty())
    {
        config_paths.push_back({fs::path(config_path), config_required});
    }
    ini::IniConfig config;
    try
    {
        config = ini::load_ini_files(config_paths);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    if (!app_config::apply_config(config, cli, options))
    {
        return 1;
    }

    auto errors = app_config::validate_options(options);
    if (!errors.empty())
    {
        std::cerr << "Config validation failed:\n";
        for (const auto &message : errors)
        {
            std::cerr << "  - " << message << "\n";
        }
        return 1;
    }

    std::cout << ">>> Working directory: " << fs::current_path() << std::endl;

    const fs::path input_path = input_resolver::resolve_input_path(options);
    if (input_path.empty() || !fs::exists(input_path))
    {
        std::cerr << "Input path not found: " << options.input_path
                  << ". Use --input to point to an image or directory." << "\n";
        return 1;
    }

    const std::string sequence_id = input_resolver::resolve_sequence_id(options, input_path);
    if (!output_from_cli)
    {
        options.output_dir = (fs::path(options.output_dir) / sequence_id).string();
    }

    const fs::path output_dir(options.output_dir);
    fs::create_directories(output_dir);
    const fs::path vis_dir = output_dir / "vis";
    if (options.save_vis)
    {
        fs::create_directories(vis_dir);
    }

    std::ofstream csv(output_dir / "detections.csv");
    if (!csv)
    {
        std::cerr << "Failed to open detections.csv for writing." << "\n";
        return 1;
    }
    output_writer::write_csv_header(csv);

    std::ofstream track_csv(output_dir / "tracks.csv");
    if (!track_csv)
    {
        std::cerr << "Failed to open tracks.csv for writing." << "\n";
        return 1;
    }
    output_writer::write_track_csv_header(track_csv);

    std::cout << ">>> Init Detector..." << std::endl;
    try
    {
        Detector2D detector(options.model_path,
                            options.score_threshold,
                            options.input_height,
                            options.input_width,
                            options.class_ids);
        SortTracker tracker(options.track_max_age,
                            options.track_min_hits,
                            options.track_iou_threshold,
                            true,
                            options.track_output_max_age);
        size_t total_detections = 0;
        int processed_frames = 0;
        int total_frames = 0;
        long long total_infer_ms = 0;

        auto process_frame = [&](const fs::path &image_file, int frame_index)
        {
            cv::Mat image = cv::imread(image_file.string());
            if (image.empty())
            {
                std::cerr << "Warning: failed to read " << image_file << "\n";
                return true;
            }

            const auto infer_start = Clock::now();
            auto detections = detector.infer(image);
            const auto infer_end = Clock::now();
            const long long infer_ms = ms_between(infer_start, infer_end);
            total_infer_ms += infer_ms;

            total_detections += detections.size();
            processed_frames++;
            output_writer::print_progress(processed_frames, total_frames);

            auto tracks = tracker.update(detections);
            output_writer::append_detections(csv, frame_index, image_file, detections);
            output_writer::append_tracks(track_csv, frame_index, image_file, tracks);

            if (options.save_vis)
            {
                cv::Mat vis = image.clone();
                draw_track_tails(vis, tracks);
                if (!tracks.empty())
                {
                    draw_tracked_objects(vis, tracks);
                }
                else
                {
                    draw_objects(vis, detections);
                }
                fs::path output_path = vis_dir / image_file.filename();
                if (!cv::imwrite(output_path.string(), vis))
                {
                    std::cerr << "Failed to write visualization: " << output_path << "\n";
                    return false;
                }
            }
            return true;
        };

        std::vector<fs::path> image_files;
        if (fs::is_regular_file(input_path))
        {
            if (!input_resolver::is_image_file(input_path))
            {
                std::cerr << "Input file is not a supported image: " << input_path << "\n";
                return 1;
            }
            image_files.push_back(input_path);
        }
        else if (fs::is_directory(input_path))
        {
            image_files = input_resolver::collect_images(input_path);
            if (image_files.empty())
            {
                std::cerr << "No images found in directory: " << input_path << "\n";
                return 1;
            }
            if (options.max_frames > 0 && options.max_frames < static_cast<int>(image_files.size()))
            {
                image_files.resize(static_cast<size_t>(options.max_frames));
            }
        }
        else
        {
            std::cerr << "Input path is neither file nor directory: " << input_path << "\n";
            return 1;
        }

        total_frames = static_cast<int>(image_files.size());
        int frame_idx = 0;
        for (const auto &image_file : image_files)
        {
            int frame_id = input_resolver::parse_frame_id(image_file, frame_idx);
            if (!process_frame(image_file, frame_id))
            {
                return 1;
            }
            frame_idx++;
        }

        if (processed_frames == 0)
        {
            std::cerr << "No frames processed." << "\n";
            return 1;
        }

        auto avg = [&](long long total)
        {
            return static_cast<double>(total) / static_cast<double>(processed_frames);
        };
        double avg_infer = avg(total_infer_ms);
        std::cout << ">>> Done. Frames: " << processed_frames
                  << ", Total detections: " << total_detections
                  << ", Avg infer: " << std::fixed << std::setprecision(2)
                  << avg_infer << " ms" << std::endl;
        std::cout << ">>> Outputs saved to: " << output_dir << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
