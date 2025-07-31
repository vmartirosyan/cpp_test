#include "fuzzy_operations.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

// --- 1. Fuzzy Logic Gate Implementations (Mamdani/Zadeh Operators) ---

double fuzzy_not(double x) {
    return 1.0 - x;
}

double fuzzy_and(double x, double y) {
    return std::min(x, y);
}

double fuzzy_or(double x, double y) {
    return std::max(x, y);
}

double fuzzy_xor(double x, double y) {
    return fuzzy_or(fuzzy_and(x, fuzzy_not(y)), fuzzy_and(fuzzy_not(x), y));
}

// --- 2. Fuzzy Full Adder (FFA) ---

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

// --- 4. Multi-Bit Fuzzy Multiplier (Bit-wise with Shifts and Additions) ---

std::vector<double> fuzzy_left_shift(const std::vector<double>& fuzzy_num, int shift_amount) {
    if (shift_amount <= 0) return fuzzy_num;
    
    std::vector<double> shifted(fuzzy_num.size() + shift_amount, 0.0);
    
    // Shift bits to the left (towards higher significance)
    for (size_t i = 0; i < fuzzy_num.size(); ++i) {
        if (i + shift_amount < shifted.size()) {
            shifted[i + shift_amount] = fuzzy_num[i];
        }
    }
    
    return shifted;
}

std::vector<double> fuzzy_multiply_basic(const std::vector<double>& FuzzyNumA,
                                        const std::vector<double>& FuzzyNumB) {
    if (FuzzyNumA.empty() || FuzzyNumB.empty()) {
        std::cerr << "Error: Cannot multiply empty fuzzy numbers." << std::endl;
        return {};
    }
    
    // Result can be up to sum of input bit widths
    int result_bits = FuzzyNumA.size() + FuzzyNumB.size();
    std::vector<double> product(result_bits, 0.0);
    
    // Shift-and-add multiplication algorithm
    for (size_t i = 0; i < FuzzyNumB.size(); ++i) {
        if (FuzzyNumB[i] > 0.0) { // Only process if multiplier bit has some membership
            // Create partial product: A * B[i] * 2^i
            std::vector<double> partial_product(FuzzyNumA.size(), 0.0);
            
            // Multiply each bit of A by the current bit of B
            for (size_t j = 0; j < FuzzyNumA.size(); ++j) {
                partial_product[j] = fuzzy_and(FuzzyNumA[j], FuzzyNumB[i]);
            }
            
            // Shift the partial product by i positions (multiply by 2^i)
            std::vector<double> shifted_product = fuzzy_left_shift(partial_product, i);
            
            // Add the shifted partial product to the accumulating result
            // Resize product to accommodate the shifted partial product if needed
            if (shifted_product.size() > product.size()) {
                product.resize(shifted_product.size(), 0.0);
            }
            
            // Ensure both vectors have the same size for addition
            size_t max_size = std::max(product.size(), shifted_product.size());
            product.resize(max_size, 0.0);
            shifted_product.resize(max_size, 0.0);
            
            // Perform fuzzy addition of partial product to result
            std::vector<double> temp_result = fuzzy_ripple_adder_basic(product, shifted_product);
            product = temp_result;
        }
    }
    
    return product;
}

std::vector<double> fuzzy_multiply_probabilistic(const std::vector<double>& FuzzyNumA,
                                                const std::vector<double>& FuzzyNumB) {
    if (FuzzyNumA.empty() || FuzzyNumB.empty()) {
        return {};
    }
    
    // Calculate crisp values and multiply them
    double crisp_A = calculate_fuzzy_value(FuzzyNumA);
    double crisp_B = calculate_fuzzy_value(FuzzyNumB);
    double crisp_product = crisp_A * crisp_B;
    
    // Convert back to fuzzy representation with uncertainty modeling
    int result_bits = FuzzyNumA.size() + FuzzyNumB.size();
    std::vector<double> fuzzy_product(result_bits, 0.0);
    
    double remaining_value = crisp_product;
    for (size_t i = 0; i < static_cast<size_t>(result_bits); ++i) {
        double bit_weight = std::pow(2, i);
        if (remaining_value >= bit_weight) {
            // Strong membership if value definitely contains this bit
            fuzzy_product[i] = std::min(1.0, remaining_value / bit_weight);
            remaining_value -= bit_weight * fuzzy_product[i];
        } else {
            // Weak membership based on fractional contribution
            fuzzy_product[i] = std::max(0.0, remaining_value / bit_weight);
        }
        
        // Add uncertainty based on input fuzziness
        // Use average uncertainty from corresponding bit positions
        double uncertainty_A = (i < FuzzyNumA.size()) ? (1.0 - FuzzyNumA[i]) : 1.0;
        double uncertainty_B = (i < FuzzyNumB.size()) ? (1.0 - FuzzyNumB[i]) : 1.0;
        double avg_uncertainty = (uncertainty_A + uncertainty_B) / 2.0;
        
        // Reduce membership based on input uncertainty (multiplication amplifies uncertainty)
        double uncertainty_factor = 1.0 - (avg_uncertainty * 0.15); // Slightly higher than addition
        fuzzy_product[i] *= uncertainty_factor;
        fuzzy_product[i] = std::max(0.0, std::min(1.0, fuzzy_product[i]));
    }
    
    return fuzzy_product;
}

std::vector<double> fuzzy_multiply_booth_inspired(const std::vector<double>& FuzzyNumA,
                                                 const std::vector<double>& FuzzyNumB) {
    if (FuzzyNumA.empty() || FuzzyNumB.empty()) {
        return {};
    }
    
    int result_bits = FuzzyNumA.size() + FuzzyNumB.size();
    std::vector<double> product(result_bits, 0.0);
    
    // Modified Booth-like approach: examine pairs of bits in multiplier
    for (size_t i = 0; i < FuzzyNumB.size(); ++i) {
        double current_bit = FuzzyNumB[i];
        double next_bit = (i + 1 < FuzzyNumB.size()) ? FuzzyNumB[i + 1] : 0.0;
        
        // Decide operation based on bit pattern (simplified)
        if (current_bit > next_bit) {
            // Add shifted multiplicand
            std::vector<double> shifted_A = fuzzy_left_shift(FuzzyNumA, i);
            if (shifted_A.size() > product.size()) {
                product.resize(shifted_A.size(), 0.0);
            }
            
            // Ensure both vectors have the same size for addition
            size_t max_size = std::max(product.size(), shifted_A.size());
            product.resize(max_size, 0.0);
            shifted_A.resize(max_size, 0.0);
            
            // Weight the addition by the difference in bit values
            double weight = current_bit - next_bit;
            for (size_t j = 0; j < shifted_A.size(); ++j) {
                shifted_A[j] *= weight;
            }
            
            std::vector<double> temp_result = fuzzy_ripple_adder_basic(product, shifted_A);
            product = temp_result;
        }
    }
    
    return product;
}

std::vector<double> fuzzy_multiply(const std::vector<double>& FuzzyNumA,
                                  const std::vector<double>& FuzzyNumB) {
    // Get results from all approaches
    std::vector<double> basic_result = fuzzy_multiply_basic(FuzzyNumA, FuzzyNumB);
    std::vector<double> prob_result = fuzzy_multiply_probabilistic(FuzzyNumA, FuzzyNumB);
    std::vector<double> booth_result = fuzzy_multiply_booth_inspired(FuzzyNumA, FuzzyNumB);
    
    // Ensure all results have the same size
    size_t max_size = std::max({basic_result.size(), prob_result.size(), booth_result.size()});
    basic_result.resize(max_size, 0.0);
    prob_result.resize(max_size, 0.0);
    booth_result.resize(max_size, 0.0);
    
    // Combine results with adaptive weighting
    std::vector<double> hybrid_result(max_size);
    
    for (size_t i = 0; i < max_size; ++i) {
        // Calculate input confidence for this bit position
        double input_conf_A = (i < FuzzyNumA.size()) ? FuzzyNumA[i] : 0.0;
        double input_conf_B = (i < FuzzyNumB.size()) ? FuzzyNumB[i] : 0.0;
        double avg_confidence = (input_conf_A + input_conf_B) / 2.0;
        
        // Adaptive weighting based on confidence and bit position
        double basic_weight = 0.4 + 0.2 * avg_confidence;  // 40-60%
        double prob_weight = 0.3 + 0.3 * avg_confidence;   // 30-60%
        double booth_weight = 0.3 - 0.1 * avg_confidence;  // 20-30%
        
        // Normalize weights
        double total_weight = basic_weight + prob_weight + booth_weight;
        basic_weight /= total_weight;
        prob_weight /= total_weight;
        booth_weight /= total_weight;
        
        hybrid_result[i] = basic_weight * basic_result[i] + 
                          prob_weight * prob_result[i] + 
                          booth_weight * booth_result[i];
        
        // Ensure result is in valid range
        hybrid_result[i] = std::max(0.0, std::min(1.0, hybrid_result[i]));
    }
    
    return hybrid_result;
}

// --- 5. Utility Functions ---

double calculate_fuzzy_value(const std::vector<double>& fuzzy_num_bits) {
    double value = 0.0;
    for (size_t i = 0; i < fuzzy_num_bits.size(); ++i) {
        value += fuzzy_num_bits[i] * std::pow(2, i);
    }
    return value;
}
