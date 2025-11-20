# -----------------------------
# Project configuration
# -----------------------------
PROJECT    := app
SRC_DIR    := src
INC_DIR    := include
BUILD_DIR  := build
TARGET     := $(BUILD_DIR)/$(PROJECT)

# -----------------------------
# Toolchain & flags
# -----------------------------
CXX        ?= g++
MODE       ?= release

WARN       := -Wall -Wextra -Wpedantic
DEPFLAGS   := -MMD -MP
INCLUDES   := -I$(INC_DIR)

ifeq ($(MODE),debug)
  OPT := -O0 -g
else
  OPT := -O2
endif

# Base compiler flags
CXXFLAGS   ?=
CXXFLAGS   += -std=c++17 $(WARN) $(OPT) $(DEPFLAGS) $(INCLUDES) -fopenmp

# Linker flags (for the main app)
LDFLAGS    ?=
LDLIBS     ?= -fopenmp

# Extra libs for tests: GoogleTest + pthread
TEST_LDFLAGS :=
TEST_LDLIBS  := $(LDLIBS) -lgtest -lpthread

# -----------------------------
# Sources / Objects / Deps
# -----------------------------
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# -----------------------------
# Tests
# -----------------------------
TEST_DIR       := tests
TEST_BUILD_DIR := $(BUILD_DIR)/tests
TEST_TARGET    := $(TEST_BUILD_DIR)/tests

TEST_SRCS := $(shell find $(TEST_DIR) -name '*.cpp')
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp,$(TEST_BUILD_DIR)/%.o,$(TEST_SRCS))
TEST_DEPS := $(TEST_OBJS:.o=.d)

# -----------------------------
# Targets
# -----------------------------
.PHONY: all clean run tree test

all: $(TARGET)

# Link final binary (application)
$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Link test binary: all test objects + all non-main app objects
$(TEST_TARGET): $(TEST_OBJS) $(filter-out $(BUILD_DIR)/$(PROJECT).o,$(OBJS))
	@mkdir -p $(dir $@)
	$(CXX) $(TEST_LDFLAGS) -o $@ $^ $(TEST_LDLIBS)

# Compile: src/xxx.cpp -> build/xxx.o (mirror directory structure)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile: tests/xxx.cpp -> build/tests/xxx.o (mirror directory structure)
$(TEST_BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Convenience
run: $(TARGET)
	./$(TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -rf $(BUILD_DIR)

# Debug helper
tree:
	@echo "MODE        = $(MODE)"
	@echo "CXX         = $(CXX)"
	@echo "SRCS        = $(SRCS)"
	@echo "OBJS        = $(OBJS)"
	@echo "TARGET      = $(TARGET)"
	@echo "TEST_SRCS   = $(TEST_SRCS)"
	@echo "TEST_OBJS   = $(TEST_OBJS)"
	@echo "TEST_TARGET = $(TEST_TARGET)"

# Auto-include header dependencies
-include $(DEPS) $(TEST_DEPS)

