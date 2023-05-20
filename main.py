import numpy as np
import matplotlib.pyplot as plt


call_count_f = 0
call_count_grad = 0


def f(x):
    global call_count_f
    call_count_f += 1
    return 4 * pow(x[0], 4) - 6 * x[0] * x[1] - 34 * x[0] + 5 * pow(x[1], 4) + 42 * x[1] + 7


def grad_f(x):
    global call_count_grad
    call_count_grad += 1
    return np.array([16*pow(x[0], 3) - 6*x[1] - 34, -6*x[0] + 20*pow(x[1], 3) + 42])


def diff_gradient():
    iter_count = 0
    x = x0.copy()
    x_traj = [x.copy()]
    while True:
        iter_count += 1
        # Обновление координат
        x_new = x - eps * grad_f(x)
        x_traj.append(x_new.copy())
        # Проверка условия остановки
        if np.linalg.norm(x_new - x) < tolerance or iter_count >= M:
            break
        x = x_new
        x_traj.append(x.copy())
    return x, iter_count, x_traj


x0 = np.array([1, 1])
eps = 0.0005
M = 1000
tolerance = 1e-6
min, iters, x_traj = diff_gradient()
print("Число итераций: ", iters)
print("Количество вычислений функции: ", call_count_f)
print("Количество вычислений градиента функции: ", call_count_grad)
print("Найденное решение (min): ", min)
print("Значение функции: ", f(min))

print("Траектория движения к экстремуму", x_traj)
# Визуализация
x = np.linspace(-2, 4, 100)
y = np.linspace(-3, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-10, 10, 100), cmap='jet')
plt.plot(*zip(*x_traj), '-o', color='black')
plt.title('Разностный аналог градиентного метода', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.show()

# import numpy as np
#
#
# call_count_f = 0
# call_count_grad = 0
#
#
# def f(x):
#     global call_count_f
#     call_count_f += 1
#     return x[0]**4 - x[0]*x[1] + x[1]**4 - 3*x[0] - 2*x[1] + 1
#
#
# def grad_f(x):
#     global call_count_grad
#     call_count_grad += 1
#     return np.array([4*x[0]**3 - x[1] - 3, 4*x[1]**3 - x[0] - 2])
#
#
# def diff_gradient():
#     iter_count = 0
#     x = x0.copy()
#     while True:
#         iter_count += 1
#         # Обновление координат
#         x_new = x - eps * grad_f(x)
#         # Проверка условия остановки
#         if np.linalg.norm(x_new - x) < tolerance or iter_count >= M:
#             break
#         x = x_new
#     return x, iter_count
#
#
# x0 = np.array([0, 1])
# eps = 0.0005
# M = 1000
# tolerance = 1e-6
# min, iters = diff_gradient()
# print("Число итераций: ", iters)
# print("Количество вычислений функции: ", call_count_f)
# print("Найденное решение (min): ", min)
# print("Значение функции: ", f(min))