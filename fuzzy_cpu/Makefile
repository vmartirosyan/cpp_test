# Compiler and flags
CXX = clang++
CXXFLAGS = -Wall -Wextra -g -std=c++17 -MD -MP #-Werror

# Directories
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build
INCLUDE_DIR = src
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Test files
TEST_FILES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TEST_FILES))

# Header files
HPP_FILES = $(wildcard $(INCLUDE_DIR)/*.hpp)

# Executable
MAIN_EXEC = $(BUILD_DIR)/simulator

# Include directories
INCLUDES = -I$(SRC_DIR) -I$(INCLUDE_DIR)

# Libraries (remove gtest if not using tests)
LIBS = 

# Default target
all: $(MAIN_EXEC)

# Create directories
$(OBJ_DIR) $(BUILD_DIR):
	mkdir -p $@

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HPP_FILES) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link main executable
$(MAIN_EXEC): $(OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

# Run the simulator
run: $(MAIN_EXEC)
	./$(MAIN_EXEC)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Include dependency files
-include $(OBJ_FILES:.o=.d)

.PHONY: all clean run
