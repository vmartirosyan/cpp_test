#include "fuzzy_cpu.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== FUZZY CPU CLASS DEMONSTRATION ===" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Create a 4-bit fuzzy CPU
    std::cout << "\n1. Creating a 4-bit Fuzzy CPU..." << std::endl;
    FuzzyCPU cpu(4, true); // verbose mode enabled
    
    // Load test data
    std::cout << "\n2. Loading fuzzy numbers into registers..." << std::endl;
    std::vector<double> fuzzy_A = {0.8, 0.6, 0.4, 0.2}; // LSB to MSB
    std::vector<double> fuzzy_B = {0.3, 0.7, 0.9, 0.5};
    
    cpu.load_register('A', fuzzy_A);
    cpu.load_register('B', fuzzy_B);
    
    std::cout << "\nCrisp values:" << std::endl;
    std::cout << "  Register A: " << std::fixed << std::setprecision(4) << cpu.get_crisp_A() << std::endl;
    std::cout << "  Register B: " << cpu.get_crisp_B() << std::endl;
    
    // Test addition using execute_instruction
    std::cout << "\n3. Testing Addition using execute_instruction('ADD')..." << std::endl;
    cpu.execute_instruction("ADD");
    std::cout << "Result crisp value: " << cpu.get_crisp_result() << std::endl;
    
    // Test multiplication using direct method call
    std::cout << "\n4. Testing Multiplication using direct multiply() method..." << std::endl;
    std::vector<double> mul_result = cpu.multiply();
    std::cout << "Result crisp value: " << FuzzyCPU::calculate_fuzzy_value(mul_result) << std::endl;
    
    // Test different patterns
    std::cout << "\n5. Testing different fuzzy patterns..." << std::endl;
    std::vector<std::string> patterns = {"uniform", "linear", "sinusoidal", "random"};
    
    for (const std::string& pattern : patterns) {
        FuzzyCPU test_cpu(3, false); // 3-bit, non-verbose
        
        std::vector<double> pattern_A = FuzzyCPU::generate_fuzzy_pattern(3, pattern, 123);
        std::vector<double> pattern_B = FuzzyCPU::generate_fuzzy_pattern(3, pattern, 456);
        
        test_cpu.load_register('A', pattern_A);
        test_cpu.load_register('B', pattern_B);
        
        // Test addition
        std::vector<double> add_result = test_cpu.add();
        double expected_add = test_cpu.get_crisp_A() + test_cpu.get_crisp_B();
        double actual_add = FuzzyCPU::calculate_fuzzy_value(add_result);
        double error_add = FuzzyCPU::calculate_relative_error(expected_add, actual_add);
        
        std::cout << "  " << std::setw(12) << pattern << " pattern ADD: " 
                  << std::fixed << std::setprecision(2) << error_add << "% error" << std::endl;
    }
    
    // Show class capabilities
    std::cout << "\n6. Class capabilities demonstration..." << std::endl;
    std::cout << "  CPU bit width: " << cpu.get_num_bits() << std::endl;
    std::cout << "  Registers loaded: " << (cpu.are_registers_loaded() ? "Yes" : "No") << std::endl;
    
    // Reset and show empty state
    cpu.reset_registers();
    std::cout << "  After reset - registers loaded: " << (cpu.are_registers_loaded() ? "Yes" : "No") << std::endl;
    
    // Test error handling
    std::cout << "\n7. Error handling test..." << std::endl;
    std::vector<double> empty_result = cpu.add(); // Should fail gracefully
    std::cout << "  Empty result size: " << empty_result.size() << " (should be 0)" << std::endl;
    
    std::cout << "\n=== DEMONSTRATION COMPLETE ===" << std::endl;
    
    return 0;
}
