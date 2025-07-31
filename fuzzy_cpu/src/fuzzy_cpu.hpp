#ifndef FUZZY_CPU_HPP
#define FUZZY_CPU_HPP

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

// Forward declarations
struct FuzzyFullAdderOutput;
class FuzzyALU;

/**
 * @brief A comprehensive fuzzy CPU simulator with fuzzy arithmetic operations
 * 
 * This class implements a conceptual CPU that operates on fuzzy numbers instead
 * of traditional binary values, demonstrating fuzzy logic applications in digital
 * arithmetic operations.
 */
class FuzzyCPU {
private:
    FuzzyALU* alu;                          ///< Arithmetic Logic Unit for fuzzy operations
    std::vector<double> reg_A;              ///< Register A for input operands
    std::vector<double> reg_B;              ///< Register B for input operands
    std::vector<double> reg_Result;         ///< Result register for operation outputs
    int num_bits;                           ///< Number of bits for CPU registers
    bool verbose;                           ///< Control output verbosity

public:
    /**
     * @brief Construct a new Fuzzy CPU object
     * @param bits Number of bits for the CPU's registers
     * @param verbose_mode Enable/disable verbose output
     */
    FuzzyCPU(int bits, bool verbose_mode = true);
    
    /**
     * @brief Destroy the Fuzzy CPU object
     */
    ~FuzzyCPU();

    // Core CPU Operations
    
    /**
     * @brief Load a fuzzy number into a register
     * @param reg_name Register name ('A' or 'B')
     * @param data Fuzzy number data (LSB to MSB format)
     */
    void load_register(char reg_name, const std::vector<double>& data);
    
    /**
     * @brief Execute a fuzzy arithmetic instruction
     * @param instruction Instruction type ("ADD" or "MUL")
     */
    void execute_instruction(const std::string& instruction);
    
    /**
     * @brief Perform fuzzy addition operation
     * @return Result of A + B as fuzzy number
     */
    std::vector<double> add();
    
    /**
     * @brief Perform fuzzy multiplication operation
     * @return Result of A * B as fuzzy number
     */
    std::vector<double> multiply();
    
    // Accessors and Utility Methods
    
    /**
     * @brief Get the result from the last operation
     * @return Reference to the result register
     */
    const std::vector<double>& get_result_register() const;
    
    /**
     * @brief Get register A contents
     * @return Reference to register A
     */
    const std::vector<double>& get_register_A() const;
    
    /**
     * @brief Get register B contents
     * @return Reference to register B
     */
    const std::vector<double>& get_register_B() const;
    
    /**
     * @brief Get crisp (defuzzified) value of register A
     * @return Crisp decimal value of register A
     */
    double get_crisp_A() const;
    
    /**
     * @brief Get crisp (defuzzified) value of register B
     * @return Crisp decimal value of register B
     */
    double get_crisp_B() const;
    
    /**
     * @brief Get crisp (defuzzified) value of result register
     * @return Crisp decimal value of result register
     */
    double get_crisp_result() const;
    
    /**
     * @brief Get number of bits in CPU registers
     * @return Number of bits
     */
    int get_num_bits() const;
    
    /**
     * @brief Set verbosity level for operations
     * @param verbose_mode Enable/disable verbose output
     */
    void set_verbose(bool verbose_mode);
    
    /**
     * @brief Reset all registers to zero
     */
    void reset_registers();
    
    /**
     * @brief Check if registers are properly loaded
     * @return True if both registers contain valid data
     */
    bool are_registers_loaded() const;
    
    // Analysis and Testing Methods
    
    /**
     * @brief Calculate relative error between expected and actual values
     * @param expected Expected value
     * @param actual Actual computed value
     * @return Relative error as percentage
     */
    static double calculate_relative_error(double expected, double actual);
    
    /**
     * @brief Print detailed error analysis for an operation
     * @param operation_name Name of the operation ("ADD" or "MUL")
     * @param expected Expected result value
     */
    void print_error_analysis(const std::string& operation_name, double expected) const;
    
    /**
     * @brief Generate fuzzy test patterns for testing
     * @param size Number of bits in the pattern
     * @param pattern Pattern type ("uniform", "linear", "sinusoidal", "random", "high_values", "low_values")
     * @param seed Seed for pattern generation
     * @return Generated fuzzy pattern
     */
    static std::vector<double> generate_fuzzy_pattern(int size, const std::string& pattern, int seed = 0);
    
    /**
     * @brief Print a fuzzy number with its crisp value
     * @param name Name/label for the fuzzy number
     * @param num_bits Fuzzy number bits (LSB to MSB)
     */
    static void print_fuzzy_number(const std::string& name, const std::vector<double>& num_bits);
    
    /**
     * @brief Calculate crisp value from fuzzy number representation
     * @param fuzzy_num_bits Fuzzy number bits (LSB to MSB)
     * @return Crisp decimal value
     */
    static double calculate_fuzzy_value(const std::vector<double>& fuzzy_num_bits);
};

#endif // FUZZY_CPU_HPP
