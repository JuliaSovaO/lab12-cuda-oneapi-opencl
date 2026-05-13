#include "integration/integrate_parallel_queue.h"
#include "config_parser/config_parser.h"
#include "functions/functions.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <stdexcept>

static double (*get_function_by_id(int func_id))(double, double) {
    switch (func_id) {
        case 1: return f1_dejong;
        case 2: return f2_ackley;
        case 3: return f3_langermann;
        default: throw std::invalid_argument("Unknown function ID");
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Wrong number of arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " func_id config_file threads points_per_task [method]" << std::endl;
        std::cerr << "  method = rectangle | simpson | montecarlo | stratified (default: montecarlo)" << std::endl;
        return 1;
    }

    try {
        int func_id = std::stoi(argv[1]);
        if (func_id < 1 || func_id > 3) {
            std::cerr << "Wrong function index" << std::endl;
            return 2;
        }

        std::string config_file = argv[2];

        int threads = std::stoi(argv[3]);
        if (threads < 1) {
            std::cerr << "Threads must be >= 1" << std::endl;
            return 64;
        }

        long long ppt = std::stoll(argv[4]);
        if (ppt <= 0) {
            std::cerr << "points_per_task must be > 0\n";
            return 64;
        }
        std::size_t points_per_task = static_cast<std::size_t>(ppt);

        std::string method = (argc == 6) ? argv[5] : "montecarlo";

        Config cfg = parse_config(config_file);
        auto func = get_function_by_id(func_id);

        IntegrationTask task{
            cfg.x_start, cfg.x_end,
            cfg.y_start, cfg.y_end,
            func_id
        };

        IntegrationResult result;

        if (method == "rectangle") {
            result = integrate_rectangle_parallel_queue(task, cfg, func, threads, points_per_task);
        } else if (method == "simpson") {
            result = integrate_simpson_parallel_queue(task, cfg, func, threads, points_per_task);
        } else if (method == "montecarlo" || method == "stratified") {
            result = integrate_monte_carlo_parallel_queue(task, cfg, func, threads, points_per_task);
        } else {
            std::cerr << "Unknown method: " << method << std::endl;
            return 64;
        }

        bool abs_ok = result.abs_error <= cfg.abs_err;
        bool rel_ok = result.rel_error <= cfg.rel_err;

        if (!abs_ok || !rel_ok) {
            std::cerr << "Required accuracy not achieved" << std::endl;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 64;
    }
}