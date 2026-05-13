#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <exception>
#include <stdexcept>

struct Config {
    double abs_err;
    double rel_err;
    double x_start;
    double x_end;
    double y_start;
    double y_end;
    int init_steps_x;
    int init_steps_y;
    int max_iter;
    std::string task_strategy;

    Config() : abs_err(0.0), rel_err(0.0), x_start(0.0), x_end(0.0),
               y_start(0.0), y_end(0.0), init_steps_x(0), init_steps_y(0),
               max_iter(0), task_strategy("uniform") {}
};

class config_open_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class config_parse_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

Config parse_config(const std::string& filename);

#endif // CONFIG_PARSER_H