import numpy as np
import matplotlib.pyplot as plt

# Вузли інтерполяції
x = np.array([-2.1, -1.1, 0.1, 1.1, 2.1])
y = np.array([-1.8647, -0.63212, 1.0, 3.7183, 9.3891])
n = len(x)

# Обчислення кроків h
h = np.diff(x)

# Формування СЛАР для других похідних (M)
A = np.zeros((n, n))
b = np.zeros(n)
A[0, 0] = 1
A[-1, -1] = 1
for i in range(1, n - 1):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]
    b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

M = np.linalg.solve(A, b)

# Значення в точці x* = -0.5
x_star = -0.5
for i in range(n - 1):
    if x[i] <= x_star <= x[i + 1]:
        idx = i
        break

h_i = x[idx + 1] - x[idx]
a = (x[idx + 1] - x_star) / h_i
b = (x_star - x[idx]) / h_i

S = (a ** 3 - a) * h_i ** 2 * M[idx] / 6 + \
    (b ** 3 - b) * h_i ** 2 * M[idx + 1] / 6 + \
    a * y[idx] + b * y[idx + 1]

print(f"f({x_star}) ≈ {S:.5f}")

# Побудова графіка
xx = np.linspace(x[0], x[-1], 300)
yy = []
for xi in xx:
    for i in range(n - 1):
        if x[i] <= xi <= x[i + 1]:
            a = (x[i + 1] - xi) / (x[i + 1] - x[i])
            b = (xi - x[i]) / (x[i + 1] - x[i])
            val = (a ** 3 - a) * (x[i + 1] - x[i]) ** 2 * M[i] / 6 + \
                  (b ** 3 - b) * (x[i + 1] - x[i]) ** 2 * M[i + 1] / 6 + \
                  a * y[i] + b * y[i + 1]
            yy.append(val)
            break

plt.figure(figsize=(8, 5))
plt.plot(xx, yy, label="Кубічний сплайн", color='blue')
plt.plot(x, y, 'o', label="Вузли інтерполяції", color='red')
plt.plot(x_star, S, 's', label=f'f({x_star}) ≈ {S:.5f}', color='green')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Кубічна сплайн-інтерполяція (власна реалізація)")
plt.grid(True)
plt.legend()
plt.show()
