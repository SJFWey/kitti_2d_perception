#include "io/input_resolver.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#include "config/app_config.h"
#include "config/ini_config.h"

namespace fs = std::filesystem;

namespace
{
bool is_numeric_string(const std::string &value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(),
                                         [](unsigned char c)
                                         { return std::isdigit(c); });
}

std::string normalize_sequence_id(std::string seq)
{
    if (is_numeric_string(seq) && seq.size() < 4)
    {
        seq.insert(seq.begin(), 4 - seq.size(), '0');
    }
    return seq;
}

bool is_image_file(const fs::path &path)
{
    std::string ext = ini::to_lower(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
}

std::vector<fs::path> collect_images(const fs::path &dir_path)
{
    std::vector<fs::path> images;
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const fs::path &file_path = entry.path();
        if (is_image_file(file_path))
        {
            images.push_back(file_path);
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

bool contains_images(const fs::path &dir_path)
{
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        if (is_image_file(entry.path()))
        {
            return true;
        }
    }
    return false;
}

std::vector<fs::path> collect_subdirs(const fs::path &dir_path)
{
    std::vector<fs::path> subdirs;
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (entry.is_directory())
        {
            subdirs.push_back(entry.path());
        }
    }
    std::sort(subdirs.begin(), subdirs.end());
    return subdirs;
}
} // namespace

namespace input_resolver
{
std::filesystem::path resolve_input_path(const app_config::AppOptions &options)
{
    if (options.input_path.empty())
    {
        return {};
    }

    const fs::path input_root(options.input_path);
    if (!fs::exists(input_root))
    {
        return {};
    }
    if (fs::is_regular_file(input_root))
    {
        return input_root;
    }
    if (contains_images(input_root))
    {
        return input_root;
    }

    // Check for KITTI tracking dataset structure: input_root/images/<sequence_id>
    fs::path images_subdir = input_root / "images";
    fs::path search_root = fs::exists(images_subdir) && fs::is_directory(images_subdir)
                               ? images_subdir
                               : input_root;

    auto sequence_dirs = collect_subdirs(search_root);
    if (sequence_dirs.empty())
    {
        return {};
    }

    if (!options.sequence_id.empty())
    {
        fs::path sequence_path = search_root / options.sequence_id;
        if (fs::exists(sequence_path))
        {
            return sequence_path;
        }
        const std::string normalized = normalize_sequence_id(options.sequence_id);
        if (normalized != options.sequence_id)
        {
            fs::path normalized_path = search_root / normalized;
            if (fs::exists(normalized_path))
            {
                return normalized_path;
            }
        }
    }

    return sequence_dirs.front();
}

std::string resolve_sequence_id(const app_config::AppOptions &options, const std::filesystem::path &input_path)
{
    if (!options.sequence_id.empty())
    {
        return normalize_sequence_id(options.sequence_id);
    }

    if (fs::is_regular_file(input_path))
    {
        return normalize_sequence_id(input_path.parent_path().filename().string());
    }

    if (fs::is_directory(input_path))
    {
        std::string candidate = input_path.filename().string();
        if (is_numeric_string(candidate))
        {
            return normalize_sequence_id(candidate);
        }
    }

    return "0000";
}

std::vector<std::filesystem::path> collect_images(const std::filesystem::path &dir_path)
{
    return ::collect_images(dir_path);
}

bool is_image_file(const std::filesystem::path &path)
{
    return ::is_image_file(path);
}

int parse_frame_id(const std::filesystem::path &image_path, int fallback)
{
    std::string stem = image_path.stem().string();
    if (is_numeric_string(stem))
    {
        return std::stoi(stem);
    }
    return fallback;
}
} // namespace input_resolver
