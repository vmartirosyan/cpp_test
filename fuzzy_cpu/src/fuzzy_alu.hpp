#ifndef FUZZY_ALU_HPP
#define FUZZY_ALU_HPP

#include <vector>
#include <iostream>

// Forward declarations
struct FuzzyFullAdderOutput;

/**
 * @brief Fuzzy Arithmetic Logic Unit for performing fuzzy arithmetic operations
 * 
 * This class implements various approaches to fuzzy arithmetic including
 * hybrid algorithms that combine multiple computation strategies for
 * improved accuracy and robustness.
 */
class FuzzyALU {
private:
    bool verbose; ///< Control output verbosity

public:
    /**
     * @brief Construct a new Fuzzy ALU object
     * @param verbose_mode Enable/disable verbose output
     */
    FuzzyALU(bool verbose_mode = true);

    /**
     * @brief Set verbosity level
     * @param verbose_mode Enable/disable verbose output
     */
    void set_verbose(bool verbose_mode);

    /**
     * @brief Performs fuzzy addition using hybrid approach
     * @param A First operand (LSB to MSB format)
     * @param B Second operand (LSB to MSB format)
     * @return Result of A + B as fuzzy number
     */
    std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B);

    /**
     * @brief Performs fuzzy multiplication using hybrid approach
     * @param A First operand (LSB to MSB format)
     * @param B Second operand (LSB to MSB format)
     * @return Result of A * B as fuzzy number
     */
    std::vector<double> multiply(const std::vector<double>& A, const std::vector<double>& B);

    // Advanced comparison methods for testing different approaches
    
    /**
     * @brief Compare different fuzzy addition approaches
     * @param A First operand
     * @param B Second operand
     * @param verbose Enable detailed output
     */
    void compare_adder_approaches(const std::vector<double>& A, const std::vector<double>& B, bool verbose = true);
    
    /**
     * @brief Compare different fuzzy multiplication approaches
     * @param A First operand
     * @param B Second operand
     * @param verbose Enable detailed output
     */
    void compare_multiplier_approaches(const std::vector<double>& A, const std::vector<double>& B, bool verbose = true);
};

#endif // FUZZY_ALU_HPP
