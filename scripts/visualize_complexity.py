import numpy as np
import matplotlib.pyplot as plt

# Input sizes
n = np.linspace(1, 2000, 2000)

# Complexity curves
y_n3 = n**3
y_n287 = n**2.87
y_n2327 = n**2.32

# Normalize all curves by the largest one (n^3 at max n)
scale = y_n3.max()
y_n3 = y_n3 / scale
y_n287 = y_n287 / scale
y_n2327 = y_n2327 / scale

plt.figure(figsize=(10, 6))

plt.plot(n, y_n3, linewidth=2, label=r"$O(n^3)$")
plt.plot(n, y_n287, linewidth=2, label=r"$O(n^{2.87}) Stassen$")
plt.plot(n, y_n2327, linewidth=2, label=r"$O(n^{2.327}) Winograd$")

plt.title("")
plt.xlabel("n (matrix dimension)")
plt.ylabel("Relative operation count (normalized)")
plt.yscale("log")

plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
