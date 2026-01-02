#include <string>

// Memory object size
#define N_MEMOBJ 3

// Gerenal Cleanup flags
#define CLEANUP_SUCCESS 0
#define CLEANUP_FAILURE 1

// General Setup flags
#define SETUP_SUCCESS 0
#define SETUP_FAILURE 1

// Context creation flags
#define CONTEXT_SUCCESS 0
#define CONTEXT_FAILURE 1

class SetupWrapper {
	public:
		~SetupWrapper();
		virtual int setup() = 0;
};
