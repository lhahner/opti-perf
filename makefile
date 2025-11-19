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
# Override from CLI if needed:
#   make MODE=debug
#   make CXX=clang++
# -----------------------------
CXX        ?= g++
MODE       ?= release

WARN       := -Wall -Wextra -Wpedantic
DEPFLAGS   := -MMD -MP
INCLUDES   := -I$(INC_DIR)

# Optimization per mode
ifeq ($(MODE),debug)
  OPT := -O0 -g
else
  OPT := -O2
endif

CXXFLAGS   ?= -std=c++17 $(WARN) $(OPT) $(DEPFLAGS) $(INCLUDES) -fopenmp
LDFLAGS    ?=
LDLIBS 	   ?= -fopenmp

# -----------------------------
# Sources / Objects / Deps
# -----------------------------
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# TESTs

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

# Link final binary
$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(TEST_TARGET): $(TEST_OBJS) $(filter-out $(BUILD_DIR)/app.o,$(OBJS))
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Compile: src/xxx.cpp -> build/xxx.o (mirror directory structure)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST_BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Convenience
run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

# Debug helper
tree:
	@echo "MODE      = $(MODE)"
	@echo "CXX       = $(CXX)"
	@echo "SRCS      = $(SRCS)"
	@echo "OBJS      = $(OBJS)"
	@echo "TARGET    = $(TARGET)"
	@echo "TEST_SRCS = $(TEST_SRCS)"
	@echo "TEST_OBJS = $(TEST_OBJS)"

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Auto-include header dependencies
-include $(DEPS) $(TEST_DEPS)

