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

/**
 * @brief Defines the common interface for setup-capable wrapper classes.
 */
class SetupWrapper {
	public:
		/**
		 * @brief Destroys the setup wrapper.
		 */
		~SetupWrapper();

		/**
		 * @brief Initializes the underlying resources of the wrapper.
		 * @return Status code indicating whether setup succeeded.
		 */
		virtual int setup() = 0;
};
