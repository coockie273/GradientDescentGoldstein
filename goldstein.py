import random

import numpy as np
import matplotlib.pyplot as plt


def quadratic_function(x, y):
    return 0.5 * x ** 2 + 0.25 * y ** 4 - 0.5 * y ** 2


def gradient_quadratic_function(x, y):
    return np.array([x, y ** 3 - y])


def gradient_descent_goldstein(f, grad_f, x0, max_iter=1000, tol=1e-5, c1=1e-4, alpha_max=1):
    x = np.array(x0, dtype=float)
    iterations = 0

    trajectory = [x.copy()]  # Траектория для графика

    while iterations < max_iter:
        gradient = np.array(grad_f(*x))
        alpha = alpha_max
        phi_alpha_0 = f(*x)
        derphi_0 = np.dot(gradient, -gradient)

        # Подбираем градиент, пока не будет удовлетворять условию Гольдштейна.
        while True:
            x_new = x - alpha * gradient
            phi_alpha = f(*x_new)

            if phi_alpha <= phi_alpha_0 + c1 * alpha * derphi_0:
                break
            else:
                alpha *= 0.5

        x -= alpha * gradient
        trajectory.append(x.copy())
        iterations += 1

        # Критерий остановки
        if np.linalg.norm(gradient) < tol:
            break

    return x, f(*x), iterations, np.array(trajectory)


# Результат
initial_point = [random.uniform(-10, 10), random.uniform(-10, 10)]
result = gradient_descent_goldstein(quadratic_function, gradient_quadratic_function, initial_point)

trajectory = result[3]
point = [round(result[0][0], 3), round(result[0][1], 3)]

# График
x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = quadratic_function(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

ax.plot(trajectory[:, 0], trajectory[:, 1], quadratic_function(trajectory[:, 0], trajectory[:, 1]), marker='o', color='red', label='Градиентный спуск')

ax.set_title(f'3D-график функции и траектории градиентного спуска.\n Минимальное значение = {round(result[1], 3)} в точке {point} ')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.legend()
plt.show()
