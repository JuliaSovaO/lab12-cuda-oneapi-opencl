#include <iostream>
#include <iomanip>
#include <cmath>

struct Config {
    double abs_err, rel_err;
    double x_start, x_end, y_start, y_end;
    int init_steps_x, init_steps_y, max_iter;
};

Config parse_config(const std::string& filename);

// CUDA function
extern "C" void run_cuda_integral(
    double x_start, double x_end,
    double y_start, double y_end,
    int steps_x, int steps_y,
    int func_id,
    double* result_value,
    double* elapsed_ms
);

double get_exact_value(int func_id) {
    switch (func_id) {
        case 1: return 4545447.652;
        case 2: return 857208.2414;
        case 3: return -1.604646665;
        default: return 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " func_id config_file steps" << std::endl;
        std::cerr << "  func_id: 1, 2, or 3" << std::endl;
        std::cerr << "  config_file: path to configuration file" << std::endl;
        std::cerr << "  steps: number of steps in each dimension (total points = steps^2)" << std::endl;
        return 1;
    }
    
    int func_id = std::stoi(argv[1]);
    if (func_id < 1 || func_id > 3) {
        std::cerr << "Wrong function index" << std::endl;
        return 2;
    }
    
    std::string config_file = argv[2];
    int steps = std::stoi(argv[3]);
    if (steps < 1) {
        std::cerr << "Steps must be >= 1" << std::endl;
        return 1;
    }
    
    Config cfg = parse_config(config_file);
    
    std::cerr << "Function: " << func_id << std::endl;
    std::cerr << "Steps: " << steps << " x " << steps << std::endl;
    std::cerr << "Range: x=[" << cfg.x_start << ", " << cfg.x_end 
              << "], y=[" << cfg.y_start << ", " << cfg.y_end << "]" << std::endl;
    
    double result;
    double elapsed_ms;
    
    run_cuda_integral(
        cfg.x_start, cfg.x_end,
        cfg.y_start, cfg.y_end,
        steps, steps,
        func_id,
        &result, &elapsed_ms
    );
    
    double exact = get_exact_value(func_id);
    double abs_error = std::abs(result - exact);
    double rel_error = abs_error / std::abs(exact);
    
    std::cout << std::setprecision(10) << result << std::endl;
    std::cout << std::setprecision(10) << abs_error << std::endl;
    std::cout << std::setprecision(10) << rel_error << std::endl;
    std::cout << (long long)elapsed_ms << std::endl;
    
    return 0;
}