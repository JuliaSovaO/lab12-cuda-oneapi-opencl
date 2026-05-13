#include "config_parser.h"
#include <fstream>
#include <algorithm>
#include <limits>
#include <cctype>
#include <string>
#include <iostream>

static std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

static double parse_double_strict(const std::string& text, int line_num, const std::string& key) {
    std::size_t pos = 0;
    try {
        double value = std::stod(text, &pos);
        if (pos != text.size()) {
            throw config_parse_error("Line " + std::to_string(line_num) +
                ": invalid floating value for '" + key + "': '" + text + "'");
        }
        return value;
    } catch (...) {
        throw config_parse_error("Line " + std::to_string(line_num) +
            ": invalid floating value for '" + key + "': '" + text + "'");
    }
}

static int parse_int_strict(const std::string& text, int line_num, const std::string& key) {
    std::size_t pos = 0;
    try {
        long long value = std::stoll(text, &pos, 10);
        if (pos != text.size()) {
            throw config_parse_error("Line " + std::to_string(line_num) +
                ": invalid integer value for '" + key + "': '" + text + "'");
        }
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
            throw config_parse_error("Line " + std::to_string(line_num) +
                ": integer value out of range for '" + key + "': '" + text + "'");
        }
        return static_cast<int>(value);
    } catch (...) {
        throw config_parse_error("Line " + std::to_string(line_num) +
            ": invalid integer value for '" + key + "': '" + text + "'");
    }
}

Config parse_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw config_open_error("Cannot open config file: " + filename);
    }

    Config cfg;
    bool seen_abs = false, seen_rel = false;
    bool seen_xs = false, seen_xe = false;
    bool seen_ys = false, seen_ye = false;
    bool seen_isx = false, seen_isy = false;
    bool seen_mi = false;
    bool seen_strategy = false;

    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        ++line_num;

        auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }

        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw config_parse_error("Line " + std::to_string(line_num) +
                ": missing '=' : '" + line + "'");
        }

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (key.empty()) {
            throw config_parse_error("Line " + std::to_string(line_num) + ": empty key");
        }
        if (val.empty()) {
            throw config_parse_error("Line " + std::to_string(line_num) +
                ": empty value for '" + key + "'");
        }

        // Parse keys
        if (key == "abs_err") {
            if (seen_abs) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'abs_err'");
            cfg.abs_err = parse_double_strict(val, line_num, key);
            seen_abs = true;
        } else if (key == "rel_err") {
            if (seen_rel) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'rel_err'");
            cfg.rel_err = parse_double_strict(val, line_num, key);
            seen_rel = true;
        } else if (key == "x_start") {
            if (seen_xs) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'x_start'");
            cfg.x_start = parse_double_strict(val, line_num, key);
            seen_xs = true;
        } else if (key == "x_end") {
            if (seen_xe) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'x_end'");
            cfg.x_end = parse_double_strict(val, line_num, key);
            seen_xe = true;
        } else if (key == "y_start") {
            if (seen_ys) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'y_start'");
            cfg.y_start = parse_double_strict(val, line_num, key);
            seen_ys = true;
        } else if (key == "y_end") {
            if (seen_ye) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'y_end'");
            cfg.y_end = parse_double_strict(val, line_num, key);
            seen_ye = true;
        } else if (key == "init_steps_x") {
            if (seen_isx) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'init_steps_x'");
            cfg.init_steps_x = parse_int_strict(val, line_num, key);
            seen_isx = true;
        } else if (key == "init_steps_y") {
            if (seen_isy) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'init_steps_y'");
            cfg.init_steps_y = parse_int_strict(val, line_num, key);
            seen_isy = true;
        } else if (key == "max_iter") {
            if (seen_mi) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'max_iter'");
            cfg.max_iter = parse_int_strict(val, line_num, key);
            seen_mi = true;
        } else if (key == "task_strategy") {
            if (seen_strategy) throw config_parse_error("Line " + std::to_string(line_num) + ": duplicate key 'task_strategy'");
            cfg.task_strategy = val;
            seen_strategy = true;
        } else {
            throw config_parse_error("Line " + std::to_string(line_num) +
                ": unknown key '" + key + "'");
        }
    }

    // check required keys
    if (!seen_abs || !seen_rel || !seen_xs || !seen_xe ||
        !seen_ys || !seen_ye || !seen_isx || !seen_isy || !seen_mi) {
        throw config_parse_error("Config missing required parameters");
    }

    if (!seen_strategy) {
        cfg.task_strategy = "uniform";
    }

    // validate values
    if (cfg.abs_err <= 0 || cfg.rel_err <= 0) {
        throw config_parse_error("abs_err and rel_err must be > 0");
    }
    if (cfg.rel_err >= 0.001) {
        throw config_parse_error("rel_err must be < 0.001");
    }
    if (cfg.x_end <= cfg.x_start) {
        throw config_parse_error("x_end must be > x_start");
    }
    if (cfg.y_end <= cfg.y_start) {
        throw config_parse_error("y_end must be > y_start");
    }
    if (cfg.init_steps_x <= 0 || cfg.init_steps_y <= 0) {
        throw config_parse_error("init_steps_x/y must be positive");
    }
    if (cfg.max_iter <= 0) {
        throw config_parse_error("max_iter must be positive");
    }

    return cfg;
}