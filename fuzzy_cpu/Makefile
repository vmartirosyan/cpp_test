# Makefile for Fuzzy CPU Simulator with Modular Architecture

# Compiler and flags
CXX = clang++
CXXFLAGS = -Wall -Wextra -g -std=c++17 -MD -MP -I$(SRC_DIR)

# Directories
SRC_DIR = src
TEST_DIR = tests
DEMO_DIR = demo
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Library source files (excluding simulator.cpp and test files)
LIB_SRC_FILES = $(SRC_DIR)/fuzzy_operations.cpp $(SRC_DIR)/fuzzy_alu.cpp $(SRC_DIR)/fuzzy_cpu.cpp
LIB_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRC_FILES))

# Original simulator source
SIM_SRC_FILES = $(SRC_DIR)/simulator.cpp
SIM_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SIM_SRC_FILES))

# Test source files
TEST_SRC_FILES = $(TEST_DIR)/test_fuzzy_cpu.cpp
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TEST_SRC_FILES))

# Demo source files
DEMO_SRC_FILES = $(DEMO_DIR)/fuzzy_cpu_demo.cpp
DEMO_OBJ_FILES = $(patsubst $(DEMO_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(DEMO_SRC_FILES))

# Header files
HPP_FILES = $(wildcard $(SRC_DIR)/*.hpp)

# Executables
SIMULATOR_EXEC = $(BUILD_DIR)/simulator
TEST_EXEC = $(BUILD_DIR)/test_fuzzy_cpu
DEMO_EXEC = $(BUILD_DIR)/fuzzy_cpu_demo

# Default target - build all
all: $(SIMULATOR_EXEC) $(TEST_EXEC) $(DEMO_EXEC)

# Build just the test framework
test: $(TEST_EXEC)

# Build just the original simulator
simulator: $(SIMULATOR_EXEC)

# Build just the demo
demo: $(DEMO_EXEC)

# Create directories
$(OBJ_DIR) $(BUILD_DIR):
	mkdir -p $@

# Compile library source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HPP_FILES) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test source files
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp $(HPP_FILES) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile demo source files
$(OBJ_DIR)/%.o: $(DEMO_DIR)/%.cpp $(HPP_FILES) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link original simulator executable
$(SIMULATOR_EXEC): $(SIM_OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Link test executable with fuzzy CPU library
$(TEST_EXEC): $(TEST_OBJ_FILES) $(LIB_OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Link demo executable with fuzzy CPU library
$(DEMO_EXEC): $(DEMO_OBJ_FILES) $(LIB_OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Run the original simulator
run: $(SIMULATOR_EXEC)
	./$(SIMULATOR_EXEC)

# Run the new test framework
run-tests: $(TEST_EXEC)
	./$(TEST_EXEC)

# Run the demo
run-demo: $(DEMO_EXEC)
	./$(DEMO_EXEC)

# Run all executables
run-all: run run-tests run-demo

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Build simulator, tests, and demo (default)"
	@echo "  simulator    - Build original simulator only"
	@echo "  test         - Build test framework only"
	@echo "  demo         - Build demo only"
	@echo "  run          - Run original simulator"
	@echo "  run-tests    - Run test framework"
	@echo "  run-demo     - Run demo"
	@echo "  run-all      - Run all executables"
	@echo "  clean        - Remove all build files"
	@echo "  help         - Show this help message"

# Include dependency files
-include $(OBJ_DIR)/*.d

# Phony targets
.PHONY: all test simulator demo clean run run-tests run-demo run-all help
