#include "config/app_config.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace
{
bool parse_int(const std::string &value, int &out)
{
    try
    {
        size_t idx = 0;
        int parsed = std::stoi(value, &idx);
        if (idx != value.size())
        {
            return false;
        }
        out = parsed;
        return true;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

bool parse_float(const std::string &value, float &out)
{
    try
    {
        size_t idx = 0;
        float parsed = std::stof(value, &idx);
        if (idx != value.size())
        {
            return false;
        }
        out = parsed;
        return true;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

bool parse_bool(const std::string &value, bool &out)
{
    const std::string lowered = ini::to_lower(ini::trim_copy(value));
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on")
    {
        out = true;
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off")
    {
        out = false;
        return true;
    }
    return false;
}

std::vector<int> parse_int_list(const std::string &value)
{
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::stringstream ss(normalized);
    std::vector<int> out;
    int v = 0;
    while (ss >> v)
    {
        out.push_back(v);
    }
    return out;
}

bool read_config_int(const ini::IniConfig &config,
                     const std::string &section,
                     const std::string &key,
                     int &out)
{
    std::string value;
    if (!ini::get_string(config, section, key, value))
    {
        return true;
    }
    int parsed = 0;
    if (!parse_int(value, parsed))
    {
        std::cerr << "Invalid config value for " << section << "." << key << ": " << value << "\n";
        return false;
    }
    out = parsed;
    return true;
}

bool read_config_float(const ini::IniConfig &config,
                       const std::string &section,
                       const std::string &key,
                       float &out)
{
    std::string value;
    if (!ini::get_string(config, section, key, value))
    {
        return true;
    }
    float parsed = 0.0f;
    if (!parse_float(value, parsed))
    {
        std::cerr << "Invalid config value for " << section << "." << key << ": " << value << "\n";
        return false;
    }
    out = parsed;
    return true;
}

bool read_config_bool(const ini::IniConfig &config,
                      const std::string &section,
                      const std::string &key,
                      bool &out)
{
    std::string value;
    if (!ini::get_string(config, section, key, value))
    {
        return true;
    }
    bool parsed = false;
    if (!parse_bool(value, parsed))
    {
        std::cerr << "Invalid config value for " << section << "." << key << ": " << value << "\n";
        return false;
    }
    out = parsed;
    return true;
}
} // namespace

namespace app_config
{
bool apply_config(const ini::IniConfig &config,
                  const CliOverrides &cli,
                  AppOptions &options)
{
    const std::string app_section = "perception2d_app";
    const std::string paths_section = "paths";

    std::string cfg_value;
    if (!cli.input_path)
    {
        if (ini::get_string(config, app_section, "input_path", cfg_value) ||
            ini::get_string(config, app_section, "input", cfg_value))
        {
            options.input_path = cfg_value;
        }
        else if (ini::get_string(config, paths_section, "kitti_tracking_root", cfg_value))
        {
            options.input_path = cfg_value;
        }
    }
    if (!cli.sequence_id)
    {
        if (ini::get_string(config, app_section, "sequence_id", cfg_value) ||
            ini::get_string(config, app_section, "sequence", cfg_value))
        {
            options.sequence_id = cfg_value;
        }
    }
    if (!cli.model_path)
    {
        if (ini::get_string(config, app_section, "model_path", cfg_value) ||
            ini::get_string(config, app_section, "model", cfg_value))
        {
            options.model_path = cfg_value;
        }
        else
        {
            std::string models_dir;
            std::string model_name;
            if (ini::get_string(config, paths_section, "models_dir", models_dir) &&
                ini::get_string(config, paths_section, "model_name", model_name))
            {
                options.model_path = (fs::path(models_dir) / model_name).string();
            }
        }
    }
    if (!cli.output_dir)
    {
        if (ini::get_string(config, app_section, "output_dir", cfg_value) ||
            ini::get_string(config, app_section, "output", cfg_value) ||
            ini::get_string(config, paths_section, "output_root", cfg_value))
        {
            options.output_dir = cfg_value;
        }
    }
    if (!cli.score_threshold)
    {
        if (!read_config_float(config, app_section, "score_threshold", options.score_threshold))
        {
            return false;
        }
    }
    if (!cli.max_frames)
    {
        if (!read_config_int(config, app_section, "max_frames", options.max_frames))
        {
            return false;
        }
    }
    if (!read_config_int(config, app_section, "input_height", options.input_height))
    {
        return false;
    }
    if (!read_config_int(config, app_section, "input_width", options.input_width))
    {
        return false;
    }
    if (ini::get_string(config, app_section, "class_ids", cfg_value))
    {
        auto parsed = parse_int_list(cfg_value);
        if (!parsed.empty())
        {
            options.class_ids = parsed;
        }
    }
    if (!read_config_int(config, app_section, "track_max_age", options.track_max_age))
    {
        return false;
    }
    if (!read_config_int(config, app_section, "track_output_max_age", options.track_output_max_age))
    {
        return false;
    }
    if (!read_config_int(config, app_section, "track_min_hits", options.track_min_hits))
    {
        return false;
    }
    if (!read_config_float(config, app_section, "track_iou_threshold", options.track_iou_threshold))
    {
        return false;
    }
    if (!cli.no_save_vis)
    {
        if (!read_config_bool(config, app_section, "save_vis", options.save_vis))
        {
            return false;
        }
    }

    if (cli.no_save_vis)
    {
        options.save_vis = false;
    }

    return true;
}

std::vector<std::string> validate_options(const AppOptions &options)
{
    std::vector<std::string> errors;
    auto add_error = [&](const std::string &message)
    {
        errors.push_back(message);
    };

    if (options.input_path.empty())
    {
        add_error("Missing required parameter: input_path. Set --input, [perception2d_app].input_path, or [paths].kitti_tracking_root.");
    }
    if (options.sequence_id.empty())
    {
        add_error("Missing required parameter: sequence_id. Set --sequence or [perception2d_app].sequence_id.");
    }
    if (options.model_path.empty())
    {
        add_error("Missing required parameter: model_path. Set --model-path, [perception2d_app].model_path, or [paths].models_dir + [paths].model_name.");
    }
    if (options.score_threshold == kUnsetFloat)
    {
        add_error("Missing business parameter: score_threshold. Set --score-threshold or [perception2d_app].score_threshold.");
    }
    else if (options.score_threshold < 0.0f || options.score_threshold > 1.0f)
    {
        add_error("score_threshold must be in [0, 1].");
    }
    if (options.input_height == kUnsetInt)
    {
        add_error("Missing business parameter: input_height. Set [perception2d_app].input_height.");
    }
    else if (options.input_height <= 0)
    {
        add_error("input_height must be positive.");
    }
    if (options.input_width == kUnsetInt)
    {
        add_error("Missing business parameter: input_width. Set [perception2d_app].input_width.");
    }
    else if (options.input_width <= 0)
    {
        add_error("input_width must be positive.");
    }
    if (options.class_ids.empty())
    {
        add_error("Missing business parameter: class_ids. Set [perception2d_app].class_ids.");
    }
    else
    {
        for (int class_id : options.class_ids)
        {
            if (class_id <= 0)
            {
                add_error("class_ids must contain positive integers.");
                break;
            }
        }
    }
    if (options.max_frames == kUnsetMaxFrames)
    {
        add_error("Missing business parameter: max_frames. Set --max-frames or [perception2d_app].max_frames.");
    }
    else if (options.max_frames == 0 || options.max_frames < -1)
    {
        add_error("max_frames must be -1 or a positive integer.");
    }
    if (options.track_max_age < 0)
    {
        if (options.track_max_age == kUnsetInt)
        {
            add_error("Missing business parameter: track_max_age. Set [perception2d_app].track_max_age.");
        }
        else
        {
            add_error("track_max_age must be >= 0.");
        }
    }
    if (options.track_output_max_age < 0)
    {
        if (options.track_output_max_age == kUnsetInt)
        {
            add_error("Missing business parameter: track_output_max_age. Set [perception2d_app].track_output_max_age.");
        }
        else
        {
            add_error("track_output_max_age must be >= 0.");
        }
    }
    if (options.track_min_hits < 0)
    {
        if (options.track_min_hits == kUnsetInt)
        {
            add_error("Missing business parameter: track_min_hits. Set [perception2d_app].track_min_hits.");
        }
        else
        {
            add_error("track_min_hits must be >= 0.");
        }
    }
    if (options.track_iou_threshold < 0.0f)
    {
        if (options.track_iou_threshold == kUnsetFloat)
        {
            add_error("Missing business parameter: track_iou_threshold. Set [perception2d_app].track_iou_threshold.");
        }
        else
        {
            add_error("track_iou_threshold must be in [0, 1].");
        }
    }
    else if (options.track_iou_threshold > 1.0f)
    {
        add_error("track_iou_threshold must be in [0, 1].");
    }

    if (!options.model_path.empty() && !fs::exists(options.model_path))
    {
        add_error("Model path not found: " + options.model_path);
    }

    if (options.output_dir.empty())
    {
        add_error("Missing output_dir. Set [perception2d_app].output_dir or [paths].output_root.");
    }

    return errors;
}
} // namespace app_config
