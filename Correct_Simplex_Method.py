import numpy as np
import matplotlib.pyplot as plt


call_count_f = 0


def f(x):
    global call_count_f
    call_count_f += 1
    return 4 * pow(x[0], 4) - 6 * x[0] * x[1] - 34 * x[0] + 5 * pow(x[1], 4) + 42 * x[1] + 7


def correct_simplex_method():
    iter_count = 0
    x_traj = []
    # Определяем начальный правильный симплекс
    n = 2  # Количество измерений
    x = np.zeros((n + 1, n))  # Матрица вершин симплекса
    for i in range(1, n + 1):
        x[i] = np.array([-1, 1])  # Начальная точка
        x_traj.append(x[i].copy())
        x[i][i - 1] = 1

    # Основной цикл алгоритма
    while True:
        iter_count += 1
        # Шаг 1. Вычисляем значения функции в вершинах и находим индекс вершины с наибольшим значением
        fx = [f(xi) for xi in x]
        j = np.argmax(fx)

        # Шаг 2. Построение нового симплекса
        # Вычисляем центр симплекса, кроме точки с наибольшим значением функции
        c = np.mean(np.delete(x, j, 0), axis=0)
        # Вычисляем новую точку
        x_new = 2 * c - x[j]
        # Вычисляем значение функции в новой точке
        f_new = f(x_new)

        # Шаг 3. Проверка условия
        # Если значение функции в новой точке меньше, чем в точке с наибольшим значением функции
        if f_new < fx[j]:
            # Заменяем точку с наибольшим значением функции на новую точку
            x[j] = x_new
            x_traj.append(x[j].copy())
        else:
            # Иначе уменьшаем размер симплекса вдвое
            x = (x + x[j]) / 2
            # Если размер симплекса стал меньше порогового значения, выходим из цикла
            if np.max(np.abs(x - x[j])) < 1e-6:
                break
    return x[j], iter_count, x_traj


min, iters, x_traj = correct_simplex_method()
# Выводим результаты
print("Число итераций: ", iters)
print("Количество вычислений функции: ", call_count_f)
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
plt.title('Метод минимизации по правильному симплексу', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.show()
