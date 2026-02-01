import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Function + gradient (2D input)
# -----------------------------
def f(x, y):
    return (x - 2.0)**2 + (y + 1.0)**2

def grad_f(x, y):
    dfdx = 2.0 * (x - 2.0)
    dfdy = 2.0 * (y + 1.0)
    return dfdx, dfdy

# -----------------------------
# Gradient Descent settings
# -----------------------------
x0, y0 = -6.0, 6.0
lr = 0.15
steps = 40

# Run GD and store trajectory
xs = [x0]
ys = [y0]
zs = [f(x0, y0)]

x, y = x0, y0
for _ in range(steps):
    gx, gy = grad_f(x, y)
    x = x - lr * gx
    y = y - lr * gy

    xs.append(x)
    ys.append(y)
    zs.append(f(x, y))

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

# -----------------------------
# Create surface mesh
# -----------------------------
x_grid = np.linspace(-8, 8, 120)
y_grid = np.linspace(-8, 8, 120)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

# -----------------------------
# Plot 3D surface + path
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# surface
ax.plot_surface(X, Y, Z, alpha=0.6, linewidth=0)

# gradient descent trajectory
ax.plot(xs, ys, zs, linewidth=2, marker="o", markersize=4, label="GD path")

ax.set_title("3D Gradient Descent Visualization")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.legend()

# -----------------------------
# Save to filesystem as PNG
# -----------------------------
out_file = "gradient_descent_3d.png"
plt.tight_layout()
plt.savefig(out_file, dpi=250)
print(f"Saved plot to: {out_file}")

