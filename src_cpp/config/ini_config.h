#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ini
{
namespace fs = std::filesystem;

struct IniConfig
{
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> sections;
};

struct IniPath
{
    fs::path path;
    bool required = false;
};

inline std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline std::string trim_copy(const std::string &value)
{
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
    {
        start++;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
        end--;
    }
    return value.substr(start, end - start);
}

inline std::string normalize_section(std::string section)
{
    if (section.empty())
    {
        return "default";
    }
    return to_lower(trim_copy(section));
}

inline void load_ini_file(const fs::path &path, IniConfig &cfg, bool required)
{
    if (path.empty())
    {
        return;
    }
    std::ifstream file(path);
    if (!file)
    {
        if (required)
        {
            throw std::runtime_error("Config not found: " + path.string());
        }
        return;
    }

    std::string line;
    std::string current_section = "default";
    while (std::getline(file, line))
    {
        const auto comment_pos = line.find_first_of("#;");
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos);
        }
        line = trim_copy(line);
        if (line.empty())
        {
            continue;
        }
        if (line.front() == '[' && line.back() == ']')
        {
            std::string section_name = trim_copy(line.substr(1, line.size() - 2));
            if (!section_name.empty())
            {
                current_section = normalize_section(section_name);
            }
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos)
        {
            continue;
        }
        std::string key = trim_copy(line.substr(0, eq));
        std::string value = trim_copy(line.substr(eq + 1));
        if (key.empty() || value.empty())
        {
            continue;
        }
        cfg.sections[current_section][to_lower(key)] = value;
    }
}

inline IniConfig load_ini_files(const std::vector<IniPath> &paths)
{
    IniConfig cfg;
    for (const auto &entry : paths)
    {
        load_ini_file(entry.path, cfg, entry.required);
    }
    return cfg;
}

inline bool get_string(const IniConfig &cfg,
                       const std::string &section,
                       const std::string &key,
                       std::string &out)
{
    const std::string section_key = normalize_section(section);
    auto section_it = cfg.sections.find(section_key);
    if (section_it == cfg.sections.end())
    {
        return false;
    }
    const std::string key_lower = to_lower(key);
    auto value_it = section_it->second.find(key_lower);
    if (value_it == section_it->second.end() || value_it->second.empty())
    {
        return false;
    }
    out = value_it->second;
    return true;
}
} // namespace ini
