#ifndef FUZZY_OPERATIONS_HPP
#define FUZZY_OPERATIONS_HPP

#include <vector>
#include <cmath>

// --- Fuzzy Logic Gate Implementations (Mamdani/Zadeh Operators) ---

/**
 * @brief Fuzzy NOT operator: 1 - x
 * @param x Input fuzzy value [0,1]
 * @return Fuzzy complement
 */
double fuzzy_not(double x);

/**
 * @brief Fuzzy AND operator: min(x, y)
 * @param x First fuzzy value [0,1]
 * @param y Second fuzzy value [0,1]
 * @return Fuzzy intersection
 */
double fuzzy_and(double x, double y);

/**
 * @brief Fuzzy OR operator: max(x, y)
 * @param x First fuzzy value [0,1]
 * @param y Second fuzzy value [0,1]
 * @return Fuzzy union
 */
double fuzzy_or(double x, double y);

/**
 * @brief Fuzzy XOR operator: (A AND NOT B) OR (NOT A AND B)
 * @param x First fuzzy value [0,1]
 * @param y Second fuzzy value [0,1]
 * @return Fuzzy exclusive or
 */
double fuzzy_xor(double x, double y);

// --- Fuzzy Full Adder Structures and Functions ---

/**
 * @brief Represents the output of a Fuzzy Full Adder: (sum_bit, carry_out)
 */
struct FuzzyFullAdderOutput {
    double sum_bit;    ///< Sum output bit
    double carry_out;  ///< Carry output bit
};

/**
 * @brief Implements a Fuzzy Full Adder using Mamdani approach
 * @param A_bit First input bit
 * @param B_bit Second input bit
 * @param Carry_in Carry input bit
 * @return Sum and carry output
 */
FuzzyFullAdderOutput fuzzy_full_adder_mamdani(double A_bit, double B_bit, double Carry_in);

/**
 * @brief Implements a Fuzzy Full Adder using probabilistic approach
 * @param A_bit First input bit
 * @param B_bit Second input bit
 * @param Carry_in Carry input bit
 * @return Sum and carry output
 */
FuzzyFullAdderOutput fuzzy_full_adder_probabilistic(double A_bit, double B_bit, double Carry_in);

/**
 * @brief Hybrid Fuzzy Full Adder combining both approaches
 * @param A_bit First input bit
 * @param B_bit Second input bit
 * @param Carry_in Carry input bit
 * @return Sum and carry output
 */
FuzzyFullAdderOutput fuzzy_full_adder(double A_bit, double B_bit, double Carry_in);

// --- Multi-Bit Fuzzy Addition Functions ---

/**
 * @brief Basic ripple-carry fuzzy adder
 * @param FuzzyNumA First operand (LSB to MSB)
 * @param FuzzyNumB Second operand (LSB to MSB)
 * @return Sum result
 */
std::vector<double> fuzzy_ripple_adder_basic(const std::vector<double>& FuzzyNumA,
                                            const std::vector<double>& FuzzyNumB);

/**
 * @brief Arithmetic approximation fuzzy adder
 * @param FuzzyNumA First operand (LSB to MSB)
 * @param FuzzyNumB Second operand (LSB to MSB)
 * @return Sum result
 */
std::vector<double> fuzzy_arithmetic_adder(const std::vector<double>& FuzzyNumA,
                                          const std::vector<double>& FuzzyNumB);

/**
 * @brief Hybrid fuzzy adder combining multiple approaches
 * @param FuzzyNumA First operand (LSB to MSB)
 * @param FuzzyNumB Second operand (LSB to MSB)
 * @return Sum result
 */
std::vector<double> fuzzy_ripple_adder(const std::vector<double>& FuzzyNumA,
                                       const std::vector<double>& FuzzyNumB);

// --- Multi-Bit Fuzzy Multiplication Functions ---

/**
 * @brief Fuzzy left shift operation (multiply by 2^n)
 * @param fuzzy_num Input fuzzy number
 * @param shift_amount Number of positions to shift left
 * @return Shifted fuzzy number
 */
std::vector<double> fuzzy_left_shift(const std::vector<double>& fuzzy_num, int shift_amount);

/**
 * @brief Basic fuzzy multiplication using shift-and-add algorithm
 * @param FuzzyNumA First operand (LSB to MSB)
 * @param FuzzyNumB Second operand (LSB to MSB)
 * @return Product result
 */
std::vector<double> fuzzy_multiply_basic(const std::vector<double>& FuzzyNumA,
                                        const std::vector<double>& FuzzyNumB);

/**
 * @brief Probabilistic approach to fuzzy multiplication
 * @param FuzzyNumA First operand (LSB to MSB)
 * @param FuzzyNumB Second operand (LSB to MSB)
 * @return Product result
 */
std::vector<double> fuzzy_multiply_probabilistic(const std::vector<double>& FuzzyNumA,
                                                const std::vector<double>& FuzzyNumB);

/**
 * @brief Booth's algorithm inspired fuzzy multiplication
 * @param FuzzyNumA First operand (LSB to MSB)
 * @param FuzzyNumB Second operand (LSB to MSB)
 * @return Product result
 */
std::vector<double> fuzzy_multiply_booth_inspired(const std::vector<double>& FuzzyNumA,
                                                 const std::vector<double>& FuzzyNumB);

/**
 * @brief Hybrid fuzzy multiplication combining multiple approaches
 * @param FuzzyNumA First operand (LSB to MSB)
 * @param FuzzyNumB Second operand (LSB to MSB)
 * @return Product result
 */
std::vector<double> fuzzy_multiply(const std::vector<double>& FuzzyNumA,
                                  const std::vector<double>& FuzzyNumB);

// --- Utility Functions ---

/**
 * @brief Calculate crisp decimal value of a multi-bit fuzzy number
 * @param fuzzy_num_bits Fuzzy number bits (LSB to MSB)
 * @return Crisp decimal value
 */
double calculate_fuzzy_value(const std::vector<double>& fuzzy_num_bits);

#endif // FUZZY_OPERATIONS_HPP
