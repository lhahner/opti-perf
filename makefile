# -----------------------------
# Project configuration
# -----------------------------
PROJECT    := app
SRC_DIR    := src
INC_DIR    := include
BUILD_DIR  := build
TARGET     := $(BUILD_DIR)/$(PROJECT)

# -----------------------------
# Torch / libtorch configuration
# -----------------------------
# Root of libtorch inside the project.
# Assumed structure:
#   lib/libtorch/include
#   lib/libtorch/lib
LIBTORCH_ROOT := $(CURDIR)/lib/libtorch

# If instead you have:
#   lib/include
#   lib/lib
# then change the line above to:
#   LIBTORCH_ROOT := $(CURDIR)/lib

TORCH_INCLUDE   := -I$(LIBTORCH_ROOT)/include -I$(LIBTORCH_ROOT)/include/torch/csrc/api/include
TORCH_LIB_DIR   := -L$(LIBTORCH_ROOT)/lib

# Libraries to link against (adjust if your setup differs)
TORCH_LIBS      := -ltorch -lc10 -ltorch_cpu -lpthread -lm

# ABI flag must match the one libtorch was built with
TORCH_CXX_FLAGS ?= -D_GLIBCXX_USE_CXX11_ABI=1

# Optional: embed runtime search path to avoid setting LD_LIBRARY_PATH
TORCH_RPATH     := -Wl,-rpath,$(LIBTORCH_ROOT)/lib

# -----------------------------
# Toolchain & flags
# -----------------------------
CXX        ?= g++
MODE       ?= release

WARN       := -Wall -Wextra -Wpedantic
DEPFLAGS   := -MMD -MP

INCLUDES   := -I$(INC_DIR) $(TORCH_INCLUDE)

ifeq ($(MODE),debug)
  OPT := -O0 -g
else
  OPT := -O2
endif

# Base compiler flags
CXXFLAGS   ?=
CXXFLAGS   += -std=c++20 $(WARN) $(OPT) $(DEPFLAGS) $(INCLUDES) -fopenmp $(TORCH_CXX_FLAGS)

# Linker flags (for the main app)
LDFLAGS    ?=
LDFLAGS    += $(TORCH_LIB_DIR) $(TORCH_RPATH)

# NOTE: fmt removed for now to avoid -lfmt error
LDLIBS     ?= $(TORCH_LIBS)

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
	@echo "MODE          = $(MODE)"
	@echo "CXX           = $(CXX)"
	@echo "SRCS          = $(SRCS)"
	@echo "OBJS          = $(OBJS)"
	@echo "TARGET        = $(TARGET)"
	@echo "TEST_SRCS     = $(TEST_SRCS)"
	@echo "TEST_OBJS     = $(TEST_OBJS)"
	@echo "TEST_TARGET   = $(TEST_TARGET)"
	@echo "LIBTORCH_ROOT = $(LIBTORCH_ROOT)"
	@echo "INCLUDES      = $(INCLUDES)"
	@echo "CXXFLAGS      = $(CXXFLAGS)"
	@echo "LDFLAGS       = $(LDFLAGS)"
	@echo "LDLIBS        = $(LDLIBS)"

# Auto-include header dependencies
-include $(DEPS) $(TEST_DEPS)

