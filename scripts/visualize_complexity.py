import numpy as np
import matplotlib.pyplot as plt

# Input sizes
n = np.linspace(1, 2000, 2000)

# Complexity curves
y_n3 = n**3
y_n287 = n**2.87
y_n2327 = n**2.327

plt.figure(figsize=(10, 6))

plt.plot(n, y_n3, linewidth=2, label=r"$O(n^3)$")
plt.plot(n, y_n287, linewidth=2, label=r"$O(n^{2.87}) Stassen$")
plt.plot(n, y_n2327, linewidth=2, label=r"$O(n^{2.327}) Winograd$")

plt.title("")
plt.xlabel("n (matrix dimension)")
plt.ylabel("Relative operation count")

plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

