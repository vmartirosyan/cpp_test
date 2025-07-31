#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "../simulator.cpp"

class FuzzyArithmeticTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test vectors
        fuzzy_a = {0.8, 0.6};  // ~2.4
        fuzzy_b = {0.5, 0.7};  // ~1.9
        fuzzy_zero = {0.0, 0.0};
        fuzzy_one = {1.0, 0.0};
    }

    std::vector<double> fuzzy_a, fuzzy_b, fuzzy_zero, fuzzy_one;
};

TEST_F(FuzzyArithmeticTest, FuzzyAdditionBasic) {
    auto result = fuzzy_addition(fuzzy_a, fuzzy_b);
    
    double expected_val = calculate_fuzzy_value(fuzzy_a) + calculate_fuzzy_value(fuzzy_b);
    double actual_val = calculate_fuzzy_value(result);
    
    EXPECT_NEAR(actual_val, expected_val, 0.1);
}

TEST_F(FuzzyArithmeticTest, FuzzyAdditionWithZero) {
    auto result = fuzzy_addition(fuzzy_a, fuzzy_zero);
    
    double expected_val = calculate_fuzzy_value(fuzzy_a);
    double actual_val = calculate_fuzzy_value(result);
    
    EXPECT_NEAR(actual_val, expected_val, 0.1);
}

TEST_F(FuzzyArithmeticTest, FuzzyMultiplicationBasic) {
    auto result = fuzzy_multiplication(fuzzy_a, fuzzy_b);
    
    double expected_val = calculate_fuzzy_value(fuzzy_a) * calculate_fuzzy_value(fuzzy_b);
    double actual_val = calculate_fuzzy_value(result);
    
    EXPECT_NEAR(actual_val, expected_val, 0.5);
}

TEST_F(FuzzyArithmeticTest, FuzzyMultiplicationWithZero) {
    auto result = fuzzy_multiplication(fuzzy_a, fuzzy_zero);
    
    double actual_val = calculate_fuzzy_value(result);
    
    EXPECT_NEAR(actual_val, 0.0, 0.1);
}

TEST_F(FuzzyArithmeticTest, FuzzyMultiplicationWithOne) {
    auto result = fuzzy_multiplication(fuzzy_a, fuzzy_one);
    
    double expected_val = calculate_fuzzy_value(fuzzy_a);
    double actual_val = calculate_fuzzy_value(result);
    
    EXPECT_NEAR(actual_val, expected_val, 0.1);
}

TEST_F(FuzzyArithmeticTest, FuzzyValueCalculation) {
    std::vector<double> test_vec = {0.5, 0.8, 0.3};
    double expected = 0.5 * 1 + 0.8 * 2 + 0.3 * 4;
    double actual = calculate_fuzzy_value(test_vec);
    
    EXPECT_DOUBLE_EQ(actual, expected);
}

TEST_F(FuzzyArithmeticTest, EmptyVectorHandling) {
    std::vector<double> empty_vec;
    double result = calculate_fuzzy_value(empty_vec);
    
    EXPECT_DOUBLE_EQ(result, 0.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}