#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace app_config
{
struct AppOptions;
}

namespace input_resolver
{
std::filesystem::path resolve_input_path(const app_config::AppOptions &options);
std::string resolve_sequence_id(const app_config::AppOptions &options, const std::filesystem::path &input_path);
std::vector<std::filesystem::path> collect_images(const std::filesystem::path &dir_path);
bool is_image_file(const std::filesystem::path &path);
int parse_frame_id(const std::filesystem::path &image_path, int fallback);
} // namespace input_resolver
