#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::min and std::max
#include <cmath>     // For std::pow
#include <iomanip>   // For std::setprecision
#include <map>       // For std::map

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations
double calculate_fuzzy_value(const std::vector<double>& fuzzy_num_bits);

// --- 1. Fuzzy Logic Gate Implementations (Mamdani/Zadeh Operators) ---

// Fuzzy NOT operator: 1 - x
double fuzzy_not(double x) {
    return 1.0 - x;
}

// Fuzzy AND operator: min(x, y)
double fuzzy_and(double x, double y) {
    return std::min(x, y);
}

// Fuzzy OR operator: max(x, y)
double fuzzy_or(double x, double y) {
    return std::max(x, y);
}

// Fuzzy XOR operator: (A AND NOT B) OR (NOT A AND B)
double fuzzy_xor(double x, double y) {
    return fuzzy_or(fuzzy_and(x, fuzzy_not(y)), fuzzy_and(fuzzy_not(x), y));
}

// --- 2. Fuzzy Full Adder (FFA) ---

// Represents the output of a Fuzzy Full Adder: (sum_bit, carry_out)
struct FuzzyFullAdderOutput {
    double sum_bit;
    double carry_out;
};

// Implements a Fuzzy Full Adder using fuzzy logic gates (Approach 1: Mamdani)
FuzzyFullAdderOutput fuzzy_full_adder_mamdani(double A_bit, double B_bit, double Carry_in) {
    FuzzyFullAdderOutput output;

    // Sum = (A XOR B) XOR Carry_in
    double sum_ab = fuzzy_xor(A_bit, B_bit);
    output.sum_bit = fuzzy_xor(sum_ab, Carry_in);

    // Carry_out = (A AND B) OR (Carry_in AND (A XOR B))
    double carry_ab = fuzzy_and(A_bit, B_bit);
    double carry_in_sum_ab = fuzzy_and(Carry_in, sum_ab);
    output.carry_out = fuzzy_or(carry_ab, carry_in_sum_ab);

    return output;
}

// Implements a Fuzzy Full Adder using probabilistic approach (Approach 2: Probabilistic)
FuzzyFullAdderOutput fuzzy_full_adder_probabilistic(double A_bit, double B_bit, double Carry_in) {
    FuzzyFullAdderOutput output;

    // Probabilistic approach: treat fuzzy values as probabilities
    // Sum bit probability: P(odd number of 1s)
    double p_one_one = A_bit * (1.0 - B_bit) * (1.0 - Carry_in) + 
                       (1.0 - A_bit) * B_bit * (1.0 - Carry_in) + 
                       (1.0 - A_bit) * (1.0 - B_bit) * Carry_in;
    double p_two_ones = A_bit * B_bit * (1.0 - Carry_in) + 
                        A_bit * (1.0 - B_bit) * Carry_in + 
                        (1.0 - A_bit) * B_bit * Carry_in;
    double p_three_ones = A_bit * B_bit * Carry_in;

    // Sum = probability of odd number of inputs being 1
    output.sum_bit = p_one_one + p_three_ones;
    
    // Carry = probability of at least two inputs being 1
    output.carry_out = p_two_ones + p_three_ones;

    // Ensure outputs are in [0,1] range
    output.sum_bit = std::max(0.0, std::min(1.0, output.sum_bit));
    output.carry_out = std::max(0.0, std::min(1.0, output.carry_out));

    return output;
}

// Hybrid Fuzzy Full Adder that combines both approaches
FuzzyFullAdderOutput fuzzy_full_adder(double A_bit, double B_bit, double Carry_in) {
    // Get results from both approaches
    FuzzyFullAdderOutput mamdani_result = fuzzy_full_adder_mamdani(A_bit, B_bit, Carry_in);
    FuzzyFullAdderOutput prob_result = fuzzy_full_adder_probabilistic(A_bit, B_bit, Carry_in);
    
    FuzzyFullAdderOutput hybrid_result;
    
    // Weighted average of both approaches
    // Give slightly more weight to probabilistic approach for better accuracy
    double mamdani_weight = 0.4;
    double prob_weight = 0.6;
    
    hybrid_result.sum_bit = mamdani_weight * mamdani_result.sum_bit + 
                           prob_weight * prob_result.sum_bit;
    hybrid_result.carry_out = mamdani_weight * mamdani_result.carry_out + 
                             prob_weight * prob_result.carry_out;
    
    return hybrid_result;
}

// --- 3. Multi-Bit Fuzzy Adder (Multiple Approaches) ---

// Original ripple-carry approach
std::vector<double> fuzzy_ripple_adder_basic(const std::vector<double>& FuzzyNumA,
                                            const std::vector<double>& FuzzyNumB) {
    if (FuzzyNumA.size() != FuzzyNumB.size()) {
        std::cerr << "Error: Fuzzy numbers must have the same number of bits for addition." << std::endl;
        return {};
    }

    int n_bits = FuzzyNumA.size();
    std::vector<double> fuzzy_sum_bits(n_bits + 1);
    double current_carry = 0.0;

    for (int i = 0; i < n_bits; ++i) {
        FuzzyFullAdderOutput result = fuzzy_full_adder(FuzzyNumA[i], FuzzyNumB[i], current_carry);
        fuzzy_sum_bits[i] = result.sum_bit;
        current_carry = result.carry_out;
    }

    fuzzy_sum_bits[n_bits] = current_carry;
    return fuzzy_sum_bits;
}

// Arithmetic approximation approach
std::vector<double> fuzzy_arithmetic_adder(const std::vector<double>& FuzzyNumA,
                                          const std::vector<double>& FuzzyNumB) {
    if (FuzzyNumA.size() != FuzzyNumB.size()) {
        std::cerr << "Error: Fuzzy numbers must have the same number of bits for addition." << std::endl;
        return {};
    }

    int n_bits = FuzzyNumA.size();
    std::vector<double> fuzzy_sum_bits(n_bits + 1, 0.0);
    
    // Calculate crisp values
    double crisp_A = calculate_fuzzy_value(FuzzyNumA);
    double crisp_B = calculate_fuzzy_value(FuzzyNumB);
    double crisp_sum = crisp_A + crisp_B;
    
    // Convert back to fuzzy representation with uncertainty modeling
    double remaining_value = crisp_sum;
    for (int i = 0; i < n_bits + 1; ++i) {
        double bit_weight = std::pow(2, i);
        if (remaining_value >= bit_weight) {
            // Strong membership if value definitely contains this bit
            fuzzy_sum_bits[i] = std::min(1.0, remaining_value / bit_weight);
            remaining_value -= bit_weight;
        } else {
            // Weak membership based on fractional contribution
            fuzzy_sum_bits[i] = std::max(0.0, remaining_value / bit_weight);
        }
        
        // Add uncertainty based on input fuzziness
        double input_uncertainty = (1.0 - FuzzyNumA[std::min(i, n_bits-1)]) + 
                                  (1.0 - FuzzyNumB[std::min(i, n_bits-1)]);
        double uncertainty_factor = 1.0 - (input_uncertainty * 0.1);
        fuzzy_sum_bits[i] *= uncertainty_factor;
        fuzzy_sum_bits[i] = std::max(0.0, std::min(1.0, fuzzy_sum_bits[i]));
    }
    
    return fuzzy_sum_bits;
}

// Improved hybrid adder combining both approaches
std::vector<double> fuzzy_ripple_adder(const std::vector<double>& FuzzyNumA,
                                       const std::vector<double>& FuzzyNumB) {
    std::vector<double> ripple_result = fuzzy_ripple_adder_basic(FuzzyNumA, FuzzyNumB);
    std::vector<double> arithmetic_result = fuzzy_arithmetic_adder(FuzzyNumA, FuzzyNumB);
    
    // Combine results with adaptive weighting
    std::vector<double> hybrid_result(ripple_result.size());
    
    for (size_t i = 0; i < hybrid_result.size(); ++i) {
        // Calculate input confidence for this bit position
        double input_conf_A = (i < FuzzyNumA.size()) ? FuzzyNumA[i] : 0.0;
        double input_conf_B = (i < FuzzyNumB.size()) ? FuzzyNumB[i] : 0.0;
        double avg_confidence = (input_conf_A + input_conf_B) / 2.0;
        
        // Higher confidence in inputs -> more weight to arithmetic approach
        // Lower confidence -> more weight to ripple approach
        double arithmetic_weight = 0.3 + 0.4 * avg_confidence;
        double ripple_weight = 1.0 - arithmetic_weight;
        
        hybrid_result[i] = ripple_weight * ripple_result[i] + 
                          arithmetic_weight * arithmetic_result[i];
        
        // Ensure result is in valid range
        hybrid_result[i] = std::max(0.0, std::min(1.0, hybrid_result[i]));
    }
    
    return hybrid_result;
}

// --- 4. Value Calculation (Defuzzification) ---

// Calculates the crisp decimal value of a multi-bit fuzzy number
// based on the sum of products of each bit value with its corresponding power of 2.
// Input vector should have LSB at index 0.
double calculate_fuzzy_value(const std::vector<double>& fuzzy_num_bits) {
    double value = 0.0;
    for (size_t i = 0; i < fuzzy_num_bits.size(); ++i) {
        value += fuzzy_num_bits[i] * std::pow(2, i);
    }
    return value;
}

// --- Helper for Printing Fuzzy Numbers ---
void print_fuzzy_number(const std::string& name, const std::vector<double>& num_bits) {
    std::cout << name << " (LSB to MSB): [";
    for (size_t i = 0; i < num_bits.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << num_bits[i];
        if (i < num_bits.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "  Crisp Value: " << std::fixed << std::setprecision(4) << calculate_fuzzy_value(num_bits) << std::endl;
}

// --- 5. Conceptual Fuzzy ALU ---

class FuzzyALU {
private:
    bool verbose; // Control output verbosity

public:
    FuzzyALU(bool verbose_mode = true) : verbose(verbose_mode) {}

    // Set verbosity level
    void set_verbose(bool verbose_mode) {
        verbose = verbose_mode;
    }

    // Performs fuzzy addition on two multi-bit fuzzy numbers using hybrid approach
    std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B) {
        if (verbose) {
            std::cout << "\n--- ALU: Performing HYBRID ADD operation ---" << std::endl;
            std::cout << "  Using combined Mamdani + Probabilistic + Arithmetic approaches" << std::endl;
            // Calculate and display the sum of crisp values
            double crisp_A = calculate_fuzzy_value(A);
            double crisp_B = calculate_fuzzy_value(B);
            double expected_crisp_sum = crisp_A + crisp_B;
            std::cout << "  Expected Crisp Sum: " << std::fixed << std::setprecision(4) << expected_crisp_sum << std::endl;
        }
        
        std::vector<double> result = fuzzy_ripple_adder(A, B);
        
        if (verbose) {
            print_fuzzy_number("Result Sum", result);
            std::cout << "----------------------------------------------" << std::endl;
        }
        return result;
    }

    // Other fuzzy operations (subtract, multiply, etc.) would go here
    // For simplicity, we only implement add for this example.
};

// --- 6. Conceptual Fuzzy CPU ---

class FuzzyCPU {
private:
    FuzzyALU alu;
    // Simple registers for demonstration
    std::vector<double> reg_A;
    std::vector<double> reg_B;
    std::vector<double> reg_Result;
    int num_bits; // Number of bits for the CPU's registers
    bool verbose; // Control output verbosity

public:
    FuzzyCPU(int bits, bool verbose_mode = true) : alu(verbose_mode), num_bits(bits), verbose(verbose_mode) {
        // Initialize registers with appropriate size and default fuzzy 0.0
        reg_A.assign(num_bits, 0.0);
        reg_B.assign(num_bits, 0.0);
        reg_Result.assign(num_bits + 1, 0.0); // Result can be one bit longer
    }

    // Set verbosity level
    void set_verbose(bool verbose_mode) {
        verbose = verbose_mode;
        alu.set_verbose(verbose_mode);
    }

    // Load a fuzzy number into a register
    void load_register(char reg_name, const std::vector<double>& data) {
        if (data.size() != static_cast<size_t>(num_bits)) {
            std::cerr << "Error: Data size mismatch for register " << reg_name << ". Expected " << num_bits << " bits." << std::endl;
            return;
        }
        if (reg_name == 'A') {
            reg_A = data;
            if (verbose) {
                std::cout << "Loaded A: "; print_fuzzy_number("", reg_A);
            }
        } else if (reg_name == 'B') {
            reg_B = data;
            if (verbose) {
                std::cout << "Loaded B: "; print_fuzzy_number("", reg_B);
            }
        } else {
            std::cerr << "Error: Invalid register name." << std::endl;
        }
    }

    // Execute a simple instruction
    void execute_instruction(const std::string& instruction) {
        if (instruction == "ADD") {
            reg_Result = alu.add(reg_A, reg_B);
        } else {
            std::cout << "Unknown instruction: " << instruction << std::endl;
        }
    }

    // Get the result from the ALU
    const std::vector<double>& get_result_register() const {
        return reg_Result;
    }

    // Get input values for analysis
    double get_crisp_A() const { return calculate_fuzzy_value(reg_A); }
    double get_crisp_B() const { return calculate_fuzzy_value(reg_B); }
    int get_num_bits() const { return num_bits; }
};

// --- Main Simulation Function ---

double calculate_relative_error(double expected, double actual) {
    if (expected == 0.0) return 0.0; // Avoid division by zero
    return std::abs((actual - expected) / expected) * 100.0; // Return as percentage
}

// Function to print error analysis after each calculation
void print_error_analysis(const std::vector<double>& result, double expected_sum) {
    double actual_sum = calculate_fuzzy_value(result);
    double rel_error = calculate_relative_error(expected_sum, actual_sum);
    
    std::cout << "  Error Analysis:" << std::endl;
    std::cout << "    Expected Sum: " << std::fixed << std::setprecision(4) << expected_sum << std::endl;
    std::cout << "    Actual Sum:   " << std::fixed << std::setprecision(4) << actual_sum << std::endl;
    std::cout << "    Relative Error: " << std::fixed << std::setprecision(6) << rel_error << "%" << std::endl;
}

// Structure to hold test results
struct TestResult {
    int bit_size;
    double expected_sum;
    double actual_sum;
    double relative_error;
    double absolute_error;
    double input_magnitude;
    double error_per_bit;
    double carry_propagation_depth;
};

// Function to generate fuzzy numbers with different patterns
std::vector<double> generate_fuzzy_pattern(int size, const std::string& pattern, int seed = 0) {
    std::vector<double> result(size);
    
    if (pattern == "uniform") {
        // Uniform distribution around 0.5
        for (int i = 0; i < size; ++i) {
            result[i] = 0.5;
        }
    } else if (pattern == "linear") {
        // Linear gradient from 0.1 to 0.9
        for (int i = 0; i < size; ++i) {
            result[i] = 0.1 + 0.8 * i / (size - 1);
        }
    } else if (pattern == "sinusoidal") {
        // Sinusoidal pattern
        for (int i = 0; i < size; ++i) {
            result[i] = 0.5 + 0.4 * std::sin(2.0 * M_PI * i / size + seed);
            result[i] = std::max(0.0, std::min(1.0, result[i]));
        }
    } else if (pattern == "random") {
        // Pseudo-random pattern based on seed
        for (int i = 0; i < size; ++i) {
            double val = 0.3 + 0.4 * std::sin(i * 0.1 + seed) + 0.2 * std::cos(i * 0.05 + seed * 2);
            result[i] = std::max(0.0, std::min(1.0, val));
        }
    } else if (pattern == "high_values") {
        // High membership values (0.7-0.9)
        for (int i = 0; i < size; ++i) {
            result[i] = 0.7 + 0.2 * std::sin(i * 0.1 + seed);
        }
    } else if (pattern == "low_values") {
        // Low membership values (0.1-0.3)
        for (int i = 0; i < size; ++i) {
            result[i] = 0.2 + 0.1 * std::sin(i * 0.1 + seed);
        }
    }
    
    return result;
}

// Function to estimate carry propagation depth
double estimate_carry_propagation(const std::vector<double>& A, const std::vector<double>& B) {
    double propagation_depth = 0.0;
    double current_carry = 0.0;
    
    for (size_t i = 0; i < A.size(); ++i) {
        // Estimate carry generation probability
        double carry_gen = fuzzy_and(A[i], B[i]);
        double carry_prop = fuzzy_xor(A[i], B[i]);
        double next_carry = fuzzy_or(carry_gen, fuzzy_and(current_carry, carry_prop));
        
        if (next_carry > 0.1) { // Threshold for significant carry
            propagation_depth += next_carry;
        }
        current_carry = next_carry;
    }
    
    return propagation_depth;
}

// Comprehensive testing function
std::vector<TestResult> run_comprehensive_tests(bool verbose = false) {
    std::vector<TestResult> results;
    
    // Test different bit sizes
    std::vector<int> bit_sizes = {2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 25, 32, 50, 64, 100, 128};
    
    // Test different patterns
    std::vector<std::string> patterns = {"uniform", "linear", "sinusoidal", "random", "high_values", "low_values"};
    
    std::cout << "\n=== COMPREHENSIVE FUZZY CPU TESTING ===" << std::endl;
    std::cout << "Testing " << bit_sizes.size() << " bit sizes with " << patterns.size() << " patterns each" << std::endl;
    std::cout << "Total tests: " << bit_sizes.size() * patterns.size() << std::endl;
    
    int test_count = 0;
    for (int bit_size : bit_sizes) {
        for (const std::string& pattern : patterns) {
            test_count++;
            if (verbose) {
                std::cout << "\n--- Test " << test_count << ": " << bit_size << "-bit, " << pattern << " pattern ---" << std::endl;
            }
            
            // Create CPU with appropriate verbosity
            FuzzyCPU cpu(bit_size, verbose);
            
            // Generate test data
            std::vector<double> fuzzy_A = generate_fuzzy_pattern(bit_size, pattern, 42);
            std::vector<double> fuzzy_B = generate_fuzzy_pattern(bit_size, pattern, 84);
            
            // Load and execute
            cpu.load_register('A', fuzzy_A);
            cpu.load_register('B', fuzzy_B);
            cpu.execute_instruction("ADD");
            
            // Calculate metrics
            double expected = cpu.get_crisp_A() + cpu.get_crisp_B();
            double actual = calculate_fuzzy_value(cpu.get_result_register());
            double rel_error = calculate_relative_error(expected, actual);
            double abs_error = std::abs(actual - expected);
            double input_magnitude = std::max(cpu.get_crisp_A(), cpu.get_crisp_B());
            double error_per_bit = rel_error / bit_size;
            double carry_depth = estimate_carry_propagation(fuzzy_A, fuzzy_B);
            
            TestResult result = {
                bit_size, expected, actual, rel_error, abs_error,
                input_magnitude, error_per_bit, carry_depth
            };
            results.push_back(result);
            
            if (verbose) {
                std::cout << "  Expected: " << std::fixed << std::setprecision(4) << expected << std::endl;
                std::cout << "  Actual: " << actual << std::endl;
                std::cout << "  Relative Error: " << std::setprecision(6) << rel_error << "%" << std::endl;
                std::cout << "  Error/Bit: " << error_per_bit << "%" << std::endl;
            }
        }
        
        // Progress indicator for non-verbose mode
        if (!verbose && bit_size % 5 == 0) {
            std::cout << "." << std::flush;
        }
    }
    
    if (!verbose) {
        std::cout << " Done!" << std::endl;
    }
    
    return results;
}

// Advanced statistical analysis
void perform_advanced_analysis(const std::vector<TestResult>& results) {
    std::cout << "\n=== ADVANCED STATISTICAL ANALYSIS ===" << std::endl;
    
    // Group results by bit size for analysis
    std::map<int, std::vector<TestResult>> grouped_results;
    for (const auto& result : results) {
        grouped_results[result.bit_size].push_back(result);
    }
    
    // 1. Error vs Bit Size Analysis
    std::cout << "\n1. ERROR SCALING ANALYSIS" << std::endl;
    std::cout << "Bit Size\tMean Error\tStd Dev\t\tMin Error\tMax Error\tError/Bit" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
    
    std::vector<int> bit_sizes;
    std::vector<double> mean_errors;
    std::vector<double> error_per_bit_means;
    
    for (const auto& group : grouped_results) {
        int bit_size = group.first;
        const auto& group_results = group.second;
        
        // Calculate statistics
        double sum_error = 0.0, sum_error_per_bit = 0.0;
        double min_error = 1e9, max_error = -1e9;
        
        for (const auto& result : group_results) {
            sum_error += result.relative_error;
            sum_error_per_bit += result.error_per_bit;
            min_error = std::min(min_error, result.relative_error);
            max_error = std::max(max_error, result.relative_error);
        }
        
        double mean_error = sum_error / group_results.size();
        double mean_error_per_bit = sum_error_per_bit / group_results.size();
        
        // Calculate standard deviation
        double sum_sq_diff = 0.0;
        for (const auto& result : group_results) {
            double diff = result.relative_error - mean_error;
            sum_sq_diff += diff * diff;
        }
        double std_dev = std::sqrt(sum_sq_diff / group_results.size());
        
        bit_sizes.push_back(bit_size);
        mean_errors.push_back(mean_error);
        error_per_bit_means.push_back(mean_error_per_bit);
        
        std::cout << std::setw(8) << bit_size 
                  << "\t" << std::fixed << std::setprecision(4) << mean_error << "%"
                  << "\t" << std::setprecision(4) << std_dev << "%"
                  << "\t\t" << std::setprecision(4) << min_error << "%"
                  << "\t" << std::setprecision(4) << max_error << "%"
                  << "\t" << std::setprecision(6) << mean_error_per_bit << "%" << std::endl;
    }
    
    // 2. Correlation Analysis
    std::cout << "\n2. CORRELATION ANALYSIS" << std::endl;
    
    // Calculate correlation between bit size and mean error
    double mean_bit_size = 0.0, mean_error_overall = 0.0;
    for (size_t i = 0; i < bit_sizes.size(); ++i) {
        mean_bit_size += bit_sizes[i];
        mean_error_overall += mean_errors[i];
    }
    mean_bit_size /= bit_sizes.size();
    mean_error_overall /= mean_errors.size();
    
    double numerator = 0.0, denom_x = 0.0, denom_y = 0.0;
    for (size_t i = 0; i < bit_sizes.size(); ++i) {
        double dx = bit_sizes[i] - mean_bit_size;
        double dy = mean_errors[i] - mean_error_overall;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }
    
    double correlation = numerator / std::sqrt(denom_x * denom_y);
    
    std::cout << "Correlation (Bit Size vs Mean Error): " << std::fixed << std::setprecision(4) << correlation << std::endl;
    
    // 3. Error Growth Rate Analysis
    std::cout << "\n3. ERROR GROWTH RATE ANALYSIS" << std::endl;
    std::cout << "Analyzing how error grows with input size..." << std::endl;
    
    // Fit a power law: error = a * size^b
    // Using log-log linear regression: log(error) = log(a) + b*log(size)
    double sum_log_size = 0.0, sum_log_error = 0.0, sum_log_size_sq = 0.0, sum_log_cross = 0.0;
    int valid_points = 0;
    
    for (size_t i = 0; i < bit_sizes.size(); ++i) {
        if (mean_errors[i] > 0) {
            double log_size = std::log(bit_sizes[i]);
            double log_error = std::log(mean_errors[i]);
            
            sum_log_size += log_size;
            sum_log_error += log_error;
            sum_log_size_sq += log_size * log_size;
            sum_log_cross += log_size * log_error;
            valid_points++;
        }
    }
    
    if (valid_points > 1) {
        double mean_log_size = sum_log_size / valid_points;
        double mean_log_error = sum_log_error / valid_points;
        
        double slope = (sum_log_cross - valid_points * mean_log_size * mean_log_error) /
                       (sum_log_size_sq - valid_points * mean_log_size * mean_log_size);
        double intercept = mean_log_error - slope * mean_log_size;
        
        std::cout << "Power law fit: Error = " << std::fixed << std::setprecision(4) 
                  << std::exp(intercept) << " * Size^" << std::setprecision(3) << slope << std::endl;
        std::cout << "Growth exponent: " << slope << " (1.0 = linear, >1 = super-linear, <1 = sub-linear)" << std::endl;
    }
    
    // 4. Pattern-Based Analysis
    std::cout << "\n4. PATTERN-BASED ERROR ANALYSIS" << std::endl;
    std::cout << "Pattern\t\tMean Error\tStd Dev\t\tMin Error\tMax Error" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    
    std::vector<std::string> patterns = {"uniform", "linear", "sinusoidal", "random", "high_values", "low_values"};
    
    for (const std::string& pattern : patterns) {
        std::vector<double> pattern_errors;
        
        for (const auto& result : results) {
            // Identify pattern by position in sequence (simple heuristic)
            int pattern_index = (&result - &results[0]) % patterns.size();
            if (patterns[pattern_index] == pattern) {
                pattern_errors.push_back(result.relative_error);
            }
        }
        
        if (!pattern_errors.empty()) {
            double sum = 0.0, min_val = 1e9, max_val = -1e9;
            for (double error : pattern_errors) {
                sum += error;
                min_val = std::min(min_val, error);
                max_val = std::max(max_val, error);
            }
            double mean = sum / pattern_errors.size();
            
            double sum_sq_diff = 0.0;
            for (double error : pattern_errors) {
                sum_sq_diff += (error - mean) * (error - mean);
            }
            double std_dev = std::sqrt(sum_sq_diff / pattern_errors.size());
            
            std::cout << std::setw(12) << pattern 
                      << "\t" << std::fixed << std::setprecision(4) << mean << "%"
                      << "\t" << std::setprecision(4) << std_dev << "%"
                      << "\t\t" << std::setprecision(4) << min_val << "%"
                      << "\t" << std::setprecision(4) << max_val << "%" << std::endl;
        }
    }
}

// Function to compare different fuzzy adder approaches
void compare_adder_approaches(const std::vector<double>& A, const std::vector<double>& B, bool verbose = true) {
    if (verbose) {
        std::cout << "\n=== ADDER APPROACH COMPARISON ===" << std::endl;
        print_fuzzy_number("Input A", A);
        print_fuzzy_number("Input B", B);
        
        double expected = calculate_fuzzy_value(A) + calculate_fuzzy_value(B);
        std::cout << "Expected crisp sum: " << std::fixed << std::setprecision(4) << expected << std::endl;
    }
    
    // Test basic ripple carry
    std::vector<double> basic_result = fuzzy_ripple_adder_basic(A, B);
    double basic_value = calculate_fuzzy_value(basic_result);
    double basic_error = calculate_relative_error(calculate_fuzzy_value(A) + calculate_fuzzy_value(B), basic_value);
    
    // Test arithmetic approach
    std::vector<double> arith_result = fuzzy_arithmetic_adder(A, B);
    double arith_value = calculate_fuzzy_value(arith_result);
    double arith_error = calculate_relative_error(calculate_fuzzy_value(A) + calculate_fuzzy_value(B), arith_value);
    
    // Test hybrid approach
    std::vector<double> hybrid_result = fuzzy_ripple_adder(A, B);
    double hybrid_value = calculate_fuzzy_value(hybrid_result);
    double hybrid_error = calculate_relative_error(calculate_fuzzy_value(A) + calculate_fuzzy_value(B), hybrid_value);
    
    if (verbose) {
        std::cout << "\nApproach Comparison:" << std::endl;
        std::cout << "  Basic Ripple:  " << std::fixed << std::setprecision(4) << basic_value 
                  << " (Error: " << std::setprecision(2) << basic_error << "%)" << std::endl;
        std::cout << "  Arithmetic:    " << std::setprecision(4) << arith_value 
                  << " (Error: " << std::setprecision(2) << arith_error << "%)" << std::endl;
        std::cout << "  Hybrid:        " << std::setprecision(4) << hybrid_value 
                  << " (Error: " << std::setprecision(2) << hybrid_error << "%)" << std::endl;
        
        std::cout << "\nBest approach: ";
        if (hybrid_error <= basic_error && hybrid_error <= arith_error) {
            std::cout << "Hybrid (chosen by default)" << std::endl;
        } else if (basic_error < arith_error) {
            std::cout << "Basic Ripple" << std::endl;
        } else {
            std::cout << "Arithmetic" << std::endl;
        }
        std::cout << "=================================" << std::endl;
    }
}

int main() {
    std::cout << "=== ENHANCED FUZZY CPU SIMULATOR WITH COMPREHENSIVE ANALYSIS ===" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Option 1: Run a few detailed examples (verbose)
    std::cout << "\n1. DETAILED EXAMPLES (Selected Test Cases)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // Create verbose examples for key bit sizes
    std::vector<int> example_sizes = {2, 4, 8, 16};
    for (int bits : example_sizes) {
        std::cout << "\n--- " << bits << "-bit Example ---" << std::endl;
        FuzzyCPU cpu(bits, true);
        
        // Use different patterns for variety
        std::vector<double> fuzzy_A = generate_fuzzy_pattern(bits, "sinusoidal", 42);
        std::vector<double> fuzzy_B = generate_fuzzy_pattern(bits, "linear", 84);
        
        cpu.load_register('A', fuzzy_A);
        cpu.load_register('B', fuzzy_B);
        cpu.execute_instruction("ADD");
        
        double expected = cpu.get_crisp_A() + cpu.get_crisp_B();
        print_error_analysis(cpu.get_result_register(), expected);
    }

    // Option 2: Run comprehensive testing (non-verbose for speed)
    std::cout << "\n\n2. COMPREHENSIVE TESTING PHASE" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    std::vector<TestResult> all_results = run_comprehensive_tests(false);

    // Option 3: Perform advanced analysis
    perform_advanced_analysis(all_results);

    // Option 4: Quick summary of original correlation analysis
    std::cout << "\n\n5. SIMPLIFIED CORRELATION SUMMARY" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    
    // Extract a subset for quick correlation display
    std::vector<int> summary_sizes = {2, 4, 8, 16, 32, 64, 100};
    std::vector<double> summary_errors;
    
    std::cout << "Bit Size\tMean Relative Error (%)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    for (int target_size : summary_sizes) {
        double sum_error = 0.0;
        int count = 0;
        
        for (const auto& result : all_results) {
            if (result.bit_size == target_size) {
                sum_error += result.relative_error;
                count++;
            }
        }
        
        if (count > 0) {
            double mean_error = sum_error / count;
            summary_errors.push_back(mean_error);
            std::cout << target_size << "\t\t" << std::fixed << std::setprecision(6) << mean_error << std::endl;
        }
    }
    
    // Quick correlation calculation for summary
    if (summary_sizes.size() == summary_errors.size() && summary_sizes.size() > 2) {
        double mean_size = 0.0, mean_error = 0.0;
        for (size_t i = 0; i < summary_sizes.size(); ++i) {
            mean_size += summary_sizes[i];
            mean_error += summary_errors[i];
        }
        mean_size /= summary_sizes.size();
        mean_error /= summary_errors.size();
        
        double numerator = 0.0, denom_x = 0.0, denom_y = 0.0;
        for (size_t i = 0; i < summary_sizes.size(); ++i) {
            double dx = summary_sizes[i] - mean_size;
            double dy = summary_errors[i] - mean_error;
            numerator += dx * dy;
            denom_x += dx * dx;
            denom_y += dy * dy;
        }
        
        double correlation = numerator / std::sqrt(denom_x * denom_y);
        std::cout << "\nQuick Correlation coefficient (r): " << std::fixed << std::setprecision(4) << correlation << std::endl;
        
        std::cout << "Interpretation: ";
        if (std::abs(correlation) > 0.8) {
            std::cout << "Strong correlation";
        } else if (std::abs(correlation) > 0.5) {
            std::cout << "Moderate correlation";
        } else if (std::abs(correlation) > 0.3) {
            std::cout << "Weak correlation";
        } else {
            std::cout << "Very weak/No correlation";
        }
        std::cout << " between input size and relative error." << std::endl;
    }

    std::cout << "\n=== ANALYSIS COMPLETE ===" << std::endl;
    std::cout << "Total tests performed: " << all_results.size() << std::endl;
    
    // Optional: Compare adder approaches on a sample input
    /*
    std::vector<double> sample_A = {0.8, 0.6, 0.4};
    std::vector<double> sample_B = {0.5, 0.7, 0.9};
    compare_adder_approaches(sample_A, sample_B);
    */

    return 0;
}