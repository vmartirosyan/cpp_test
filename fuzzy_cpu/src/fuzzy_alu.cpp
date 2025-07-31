#include "fuzzy_alu.hpp"
#include "fuzzy_operations.hpp"
#include <iomanip>

FuzzyALU::FuzzyALU(bool verbose_mode) : verbose(verbose_mode) {}

void FuzzyALU::set_verbose(bool verbose_mode) {
    verbose = verbose_mode;
}

std::vector<double> FuzzyALU::add(const std::vector<double>& A, const std::vector<double>& B) {
    if (verbose) {
        std::cout << "\n--- ALU: Performing ADAPTIVE ADD operation ---" << std::endl;
        std::cout << "  Using algorithm selection based on input characteristics" << std::endl;
        // Calculate and display the sum of crisp values
        double crisp_A = calculate_fuzzy_value(A);
        double crisp_B = calculate_fuzzy_value(B);
        double expected_crisp_sum = crisp_A + crisp_B;
        std::cout << "  Expected Crisp Sum: " << std::fixed << std::setprecision(4) << expected_crisp_sum << std::endl;
    }
    
    // Adaptive algorithm selection for addition
    std::vector<double> result;
    
    // Calculate input confidence
    double avg_conf_A = 0.0, avg_conf_B = 0.0;
    for (double val : A) avg_conf_A += val;
    for (double val : B) avg_conf_B += val;
    avg_conf_A /= A.size();
    avg_conf_B /= B.size();
    double avg_confidence = (avg_conf_A + avg_conf_B) / 2.0;
    
    // Choose algorithm based on bit size and input confidence
    if (A.size() <= 4 || avg_confidence < 0.4) {
        // For small bit sizes or low confidence, use basic ripple
        result = fuzzy_ripple_adder_basic(A, B);
        if (verbose) std::cout << "  Selected: Basic Ripple (small/low-confidence)" << std::endl;
    } else {
        // For larger bit sizes with good confidence, use hybrid
        result = fuzzy_ripple_adder(A, B);
        if (verbose) std::cout << "  Selected: Hybrid (large/high-confidence)" << std::endl;
    }
    
    if (verbose) {
        std::cout << "Result Sum (LSB to MSB): [";
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << result[i];
            if (i < result.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        std::cout << "  Crisp Value: " << std::fixed << std::setprecision(4) 
                  << calculate_fuzzy_value(result) << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
    }
    return result;
}

std::vector<double> FuzzyALU::multiply(const std::vector<double>& A, const std::vector<double>& B) {
    if (verbose) {
        std::cout << "\n--- ALU: Performing ADAPTIVE MULTIPLY operation ---" << std::endl;
        std::cout << "  Using algorithm selection based on input characteristics" << std::endl;
        // Calculate and display the product of crisp values
        double crisp_A = calculate_fuzzy_value(A);
        double crisp_B = calculate_fuzzy_value(B);
        double expected_crisp_product = crisp_A * crisp_B;
        std::cout << "  Expected Crisp Product: " << std::fixed << std::setprecision(4) << expected_crisp_product << std::endl;
    }
    
    // Adaptive algorithm selection for multiplication
    std::vector<double> result;
    
    // Calculate input magnitude and confidence
    double crisp_A = calculate_fuzzy_value(A);
    double crisp_B = calculate_fuzzy_value(B);
    double magnitude = std::max(crisp_A, crisp_B);
    
    // Calculate average confidence
    double avg_conf_A = 0.0, avg_conf_B = 0.0;
    for (double val : A) avg_conf_A += val;
    for (double val : B) avg_conf_B += val;
    avg_conf_A /= A.size();
    avg_conf_B /= B.size();
    double avg_confidence = (avg_conf_A + avg_conf_B) / 2.0;
    
    // Choose algorithm based on characteristics
    if (A.size() <= 6 || magnitude < 10.0 || avg_confidence < 0.5) {
        // For small operations, use probabilistic
        result = fuzzy_multiply_probabilistic(A, B);
        if (verbose) std::cout << "  Selected: Probabilistic (small/simple)" << std::endl;
    } else {
        // For larger operations, use hybrid for better stability
        result = fuzzy_multiply(A, B);
        if (verbose) std::cout << "  Selected: Hybrid (large/complex)" << std::endl;
    }
    
    if (verbose) {
        std::cout << "Result Product (LSB to MSB): [";
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << result[i];
            if (i < result.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        std::cout << "  Crisp Value: " << std::fixed << std::setprecision(4) 
                  << calculate_fuzzy_value(result) << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
    }
    return result;
}

void FuzzyALU::compare_adder_approaches(const std::vector<double>& A, const std::vector<double>& B, bool verbose_output) {
    if (verbose_output) {
        std::cout << "\n=== ADDER APPROACH COMPARISON ===" << std::endl;
        std::cout << "Input A (LSB to MSB): [";
        for (size_t i = 0; i < A.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << A[i];
            if (i < A.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Crisp Value: " << std::fixed << std::setprecision(4) << calculate_fuzzy_value(A) << std::endl;
        
        std::cout << "Input B (LSB to MSB): [";
        for (size_t i = 0; i < B.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << B[i];
            if (i < B.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Crisp Value: " << std::fixed << std::setprecision(4) << calculate_fuzzy_value(B) << std::endl;
        
        double expected = calculate_fuzzy_value(A) + calculate_fuzzy_value(B);
        std::cout << "Expected crisp sum: " << std::fixed << std::setprecision(4) << expected << std::endl;
    }
    
    // Calculate relative error helper
    auto calc_error = [](double expected, double actual) -> double {
        if (expected == 0.0) return 0.0;
        return std::abs((actual - expected) / expected) * 100.0;
    };
    
    double expected = calculate_fuzzy_value(A) + calculate_fuzzy_value(B);
    
    // Test basic ripple carry
    std::vector<double> basic_result = fuzzy_ripple_adder_basic(A, B);
    double basic_value = calculate_fuzzy_value(basic_result);
    double basic_error = calc_error(expected, basic_value);
    
    // Test arithmetic approach
    std::vector<double> arith_result = fuzzy_arithmetic_adder(A, B);
    double arith_value = calculate_fuzzy_value(arith_result);
    double arith_error = calc_error(expected, arith_value);
    
    // Test hybrid approach
    std::vector<double> hybrid_result = fuzzy_ripple_adder(A, B);
    double hybrid_value = calculate_fuzzy_value(hybrid_result);
    double hybrid_error = calc_error(expected, hybrid_value);
    
    if (verbose_output) {
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

void FuzzyALU::compare_multiplier_approaches(const std::vector<double>& A, const std::vector<double>& B, bool verbose_output) {
    if (verbose_output) {
        std::cout << "\n=== MULTIPLIER APPROACH COMPARISON ===" << std::endl;
        std::cout << "Input A (LSB to MSB): [";
        for (size_t i = 0; i < A.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << A[i];
            if (i < A.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Crisp Value: " << std::fixed << std::setprecision(4) << calculate_fuzzy_value(A) << std::endl;
        
        std::cout << "Input B (LSB to MSB): [";
        for (size_t i = 0; i < B.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << B[i];
            if (i < B.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Crisp Value: " << std::fixed << std::setprecision(4) << calculate_fuzzy_value(B) << std::endl;
        
        double expected = calculate_fuzzy_value(A) * calculate_fuzzy_value(B);
        std::cout << "Expected crisp product: " << std::fixed << std::setprecision(4) << expected << std::endl;
    }
    
    // Calculate relative error helper
    auto calc_error = [](double expected, double actual) -> double {
        if (expected == 0.0) return 0.0;
        return std::abs((actual - expected) / expected) * 100.0;
    };
    
    double expected = calculate_fuzzy_value(A) * calculate_fuzzy_value(B);
    
    // Test basic shift-add multiplication
    std::vector<double> basic_result = fuzzy_multiply_basic(A, B);
    double basic_value = calculate_fuzzy_value(basic_result);
    double basic_error = calc_error(expected, basic_value);
    
    // Test probabilistic approach
    std::vector<double> prob_result = fuzzy_multiply_probabilistic(A, B);
    double prob_value = calculate_fuzzy_value(prob_result);
    double prob_error = calc_error(expected, prob_value);
    
    // Test booth-inspired approach
    std::vector<double> booth_result = fuzzy_multiply_booth_inspired(A, B);
    double booth_value = calculate_fuzzy_value(booth_result);
    double booth_error = calc_error(expected, booth_value);
    
    // Test hybrid approach
    std::vector<double> hybrid_result = fuzzy_multiply(A, B);
    double hybrid_value = calculate_fuzzy_value(hybrid_result);
    double hybrid_error = calc_error(expected, hybrid_value);
    
    if (verbose_output) {
        std::cout << "\nMultiplier Approach Comparison:" << std::endl;
        std::cout << "  Shift-Add:     " << std::fixed << std::setprecision(4) << basic_value 
                  << " (Error: " << std::setprecision(2) << basic_error << "%)" << std::endl;
        std::cout << "  Probabilistic: " << std::setprecision(4) << prob_value 
                  << " (Error: " << std::setprecision(2) << prob_error << "%)" << std::endl;
        std::cout << "  Booth-inspired:" << std::setprecision(4) << booth_value 
                  << " (Error: " << std::setprecision(2) << booth_error << "%)" << std::endl;
        std::cout << "  Hybrid:        " << std::setprecision(4) << hybrid_value 
                  << " (Error: " << std::setprecision(2) << hybrid_error << "%)" << std::endl;
        
        std::cout << "\nBest approach: ";
        if (hybrid_error <= basic_error && hybrid_error <= prob_error && hybrid_error <= booth_error) {
            std::cout << "Hybrid (chosen by default)" << std::endl;
        } else if (basic_error <= prob_error && basic_error <= booth_error) {
            std::cout << "Shift-Add" << std::endl;
        } else if (prob_error <= booth_error) {
            std::cout << "Probabilistic" << std::endl;
        } else {
            std::cout << "Booth-inspired" << std::endl;
        }
        std::cout << "====================================" << std::endl;
    }
}
