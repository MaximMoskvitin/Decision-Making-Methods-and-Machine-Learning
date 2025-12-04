import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def mse_loss(y_true, y_pred):
    """
    Среднеквадратичная ошибка: MSE = mean((y_true - y_pred)^2)
    """
    return tf.reduce_mean((y_true - y_pred) ** 2)


def create_quadratic_dataset(m, min_value, max_value,
                             k1_true, k0_true, b_true,
                             noise_std=1.0):
    """
    Создаём данные для квадратичной регрессии:
    x ~ U[min_value, max_value]
    N ~ N(0, noise_std^2)
    y = k1_true * x^2 + k0_true * x + b_true + N
    """
    # тензор равномерно распределённых случайных величин
    x = tf.random.uniform(shape=(m,),
                          minval=min_value,
                          maxval=max_value)

    # тензор нормально распределённых случайных величин
    noise = tf.random.normal(shape=(m,),
                             mean=0.0,
                             stddev=noise_std)

    y = k1_true * x ** 2 + k0_true * x + b_true + noise

    return x, noise, y


def create_linear_dataset(m, min_value, max_value,
                          k0_true, b_true,
                          noise_std=1.0):
    """
    Создаём данные для линейной регрессии:
    x ~ U[min_value, max_value]
    N ~ N(0, noise_std^2)
    y = k0_true * x + b_true + N
    """
    x = tf.random.uniform(shape=(m,),
                          minval=min_value,
                          maxval=max_value)

    noise = tf.random.normal(shape=(m,),
                             mean=0.0,
                             stddev=noise_std)

    y = k0_true * x + b_true + noise

    return x, noise, y


def plot_loss(loss_history, title):
    """
    График изменения значения функции потерь по эпохам.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_quadratic_fit(x, y,
                       k1_true, k0_true, b_true,
                       k1_learned, k0_learned, b_learned,
                       title):
    """
    Визуализация: точки выборки, истинный полином, найденный полином.
    """
    x_np = x.numpy()
    y_np = y.numpy()

    xs = np.linspace(x_np.min() - 0.5,
                     x_np.max() + 0.5,
                     400)

    # истинный полином
    ys_true = k1_true * xs ** 2 + k0_true * xs + b_true

    # полином, полученный после обучения
    ys_model = k1_learned * xs ** 2 + k0_learned * xs + b_learned

    plt.figure(figsize=(6, 4))
    plt.scatter(x_np, y_np, alpha=0.5, label="данные")
    plt.plot(xs, ys_true, '--', label="истинный полином", linewidth=2)
    plt.plot(xs, ys_model, label="найденный полином", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_linear_fit(x, y,
                    k0_true, b_true,
                    k0_learned, b_learned,
                    title):
    """
    Визуализация: точки выборки, истинная прямая, найденная прямая.
    """
    x_np = x.numpy()
    y_np = y.numpy()

    xs = np.linspace(x_np.min() - 0.5,
                     x_np.max() + 0.5,
                     400)

    ys_true = k0_true * xs + b_true
    ys_model = k0_learned * xs + b_learned

    plt.figure(figsize=(6, 4))
    plt.scatter(x_np, y_np, alpha=0.5, label="данные")
    plt.plot(xs, ys_true, '--', label="истинная прямая", linewidth=2)
    plt.plot(xs, ys_model, label="найденная прямая", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


# ==========================
# ГРАДИЕНТНЫЙ СПУСК: ОБУЧЕНИЕ
# ==========================

def train_quadratic_regression(x, y,
                               learning_rate=0.01,
                               epochs=1000):
    """
    Обучение квадратичной регрессии методом градиентного спуска.
    Модель: y_hat = k1 * x^2 + k0 * x + b
    """
    # инициализация весов модели как тензоров tf.Variable
    k1 = tf.Variable(tf.random.normal(shape=()),
                     name="k1")
    k0 = tf.Variable(tf.random.normal(shape=()),
                     name="k0")
    b = tf.Variable(tf.random.normal(shape=()),
                    name="b")

    # оптимизатор градиентного спуска
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    loss_history = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = k1 * x ** 2 + k0 * x + b
            loss = mse_loss(y, y_pred)

        # вычисляем градиенты по трём параметрам
        grads = tape.gradient(loss, [k1, k0, b])

        # делаем шаг градиентного спуска
        optimizer.apply_gradients(zip(grads, [k1, k0, b]))

        loss_history.append(float(loss.numpy()))

        # выводим промежуточную информацию каждые 100 эпох
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: loss = {loss.numpy():.4f}", end='\n')
            print(f"  k1 = {k1.numpy():.4f}, k0 = {k0.numpy():.4f}, b = {b.numpy():.4f}", end='\n\n')

    return k1.numpy(), k0.numpy(), b.numpy(), loss_history


def train_linear_regression(x, y,
                            learning_rate=0.01,
                            epochs=500):
    """
    Обучение линейной регрессии методом градиентного спуска.
    Модель: y_hat = k0 * x + b
    """
    k0 = tf.Variable(tf.random.normal(shape=()),
                     name="k0")
    b = tf.Variable(tf.random.normal(shape=()),
                    name="b")

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    loss_history = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = k0 * x + b
            loss = mse_loss(y, y_pred)

        grads = tape.gradient(loss, [k0, b])
        optimizer.apply_gradients(zip(grads, [k0, b]))

        loss_history.append(float(loss.numpy()))

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: loss = {loss.numpy():.4f}", end='\n')
            print(f"  k0 = {k0.numpy():.4f}, b = {b.numpy():.4f}", end='\n\n')

    return k0.numpy(), b.numpy(), loss_history


# ==========================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================

def main():
    # фиксируем seed'ы для воспроизводимости
    tf.random.set_seed(42)
    np.random.seed(42)

    # ----------------------
    # ЧАСТЬ 1. КВАДРАТИЧНАЯ РЕГРЕССИЯ
    # ----------------------
    print("=== ЧАСТЬ 1. КВАДРАТИЧНАЯ РЕГРЕССИЯ ===", end='\n\n')

    m = 200
    min_value = -3.0
    max_value = 3.0
    noise_std = 1.0

    # истинные параметры модели
    k1_true = 0.7
    k0_true = -1.2
    b_true = 2.0

    # создаём искусственные данные
    x_quad, noise_quad, y_quad = create_quadratic_dataset(
        m=m,
        min_value=min_value,
        max_value=max_value,
        k1_true=k1_true,
        k0_true=k0_true,
        b_true=b_true,
        noise_std=noise_std
    )

    print("Примеры значений тензоров для квадратичной регрессии:", end='\n')
    print("x_quad shape:", x_quad.shape, end='\n')
    print("noise_quad shape:", noise_quad.shape, end='\n')
    print("y_quad shape:", y_quad.shape, end='\n\n')

    # визуализируем исходную зависимость
    plt.figure(figsize=(6, 4))
    plt.scatter(x_quad.numpy(), y_quad.numpy(), alpha=0.5, label="данные")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Искусственные данные (квадратичная регрессия)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # обучаем модель методом градиентного спуска
    k1_learned, k0_learned, b_learned, loss_history_quad = train_quadratic_regression(
        x_quad,
        y_quad,
        learning_rate=0.01,
        epochs=1000
    )

    # вывод сравнения параметров
    print("Истинные параметры квадратичной модели:", end='\n')
    print(f"  k1_true = {k1_true:.4f}, k0_true = {k0_true:.4f}, b_true = {b_true:.4f}", end='\n\n')

    print("Найденные параметры квадратичной модели:", end='\n')
    print(f"  k1 = {k1_learned:.4f}, k0 = {k0_learned:.4f}, b = {b_learned:.4f}", end='\n\n')

    # график функции потерь
    plot_loss(loss_history_quad,
              title="Функция потерь (квадратичная регрессия)")

    # визуализация полиномов
    plot_quadratic_fit(
        x_quad,
        y_quad,
        k1_true, k0_true, b_true,
        k1_learned, k0_learned, b_learned,
        title="Аппроксимация квадратичной зависимости"
    )

    # ----------------------
    # ЧАСТЬ 2. ЛИНЕЙНАЯ РЕГРЕССИЯ
    # ----------------------
    print("=== ЧАСТЬ 2. ЛИНЕЙНАЯ РЕГРЕССИЯ ===", end='\n\n')

    m_lin = 200
    min_value_lin = -5.0
    max_value_lin = 5.0
    noise_std_lin = 1.0

    k0_true_lin = 1.5
    b_true_lin = -0.5

    x_lin, noise_lin, y_lin = create_linear_dataset(
        m=m_lin,
        min_value=min_value_lin,
        max_value=max_value_lin,
        k0_true=k0_true_lin,
        b_true=b_true_lin,
        noise_std=noise_std_lin
    )

    print("Примеры значений тензоров для линейной регрессии:", end='\n')
    print("x_lin shape:", x_lin.shape, end='\n')
    print("noise_lin shape:", noise_lin.shape, end='\n')
    print("y_lin shape:", y_lin.shape, end='\n\n')

    plt.figure(figsize=(6, 4))
    plt.scatter(x_lin.numpy(), y_lin.numpy(), alpha=0.5, label="данные")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Искусственные данные (линейная регрессия)")
    plt.grid(True)
    plt.legend()
    plt.show()

    k0_learned_lin, b_learned_lin, loss_history_lin = train_linear_regression(
        x_lin,
        y_lin,
        learning_rate=0.01,
        epochs=500
    )

    print("Истинные параметры линейной модели:", end='\n')
    print(f"  k0_true = {k0_true_lin:.4f}, b_true = {b_true_lin:.4f}", end='\n\n')

    print("Найденные параметры линейной модели:", end='\n')
    print(f"  k0 = {k0_learned_lin:.4f}, b = {b_learned_lin:.4f}", end='\n\n')

    plot_loss(loss_history_lin,
              title="Функция потерь (линейная регрессия)")

    plot_linear_fit(
        x_lin,
        y_lin,
        k0_true_lin, b_true_lin,
        k0_learned_lin, b_learned_lin,
        title="Аппроксимация линейной зависимости"
    )


if __name__ == "__main__":
    main()

'''
1. Что такое градиентный спуск и как он используется в машинном обучении?
Градиентный спуск — это итерационный метод поиска минимума функции.
В машинном обучении этой функцией обычно является функция потерь, 
которая измеряет, насколько сильно предсказания модели отличаются от правильных ответов.
Идея метода:
    1) считаем значение функции потерь на текущих весах модели
    2) вычисляем градиент (направление наибольшего роста функции)
    3) делаем шаг в противоположную сторону (по антиградиенту), чтобы уменьшить значение функции потерь
    4) повторяем шаги до тех пор, пока ошибка не станет достаточно маленькой
       или пока не закончится количество эпох обучения.
Так модель постепенно «учится» и подбирает такие веса, при которых
функция потерь минимальна или близка к минимуму.


2. Что такое градиент в контексте градиентного спуска?
Градиент — это вектор из частных производных функции по всем её параметрам.
Если представить, что у нас есть функция потерь F(w1, w2, ..., wn),
то градиент ∇F — это набор производных dF/dw1, dF/dw2, ..., dF/dwn.
Важные моменты:
    - градиент указывает направление наиболее быстрого роста функции;
    - если двигаться в сторону, противоположную градиенту (по антиградиенту), то мы идём к уменьшению значения функции.
В градиентном спуске мы как раз используем антиградиент, чтобы уменьшать функцию потерь и улучшать качество модели.


3. Как обновляются веса в градиентном спуске?
Веса обновляются по формуле:
    w_{new} = w_{old} - η * ∇F(w_{old})
где:
    w_{old}  — старые значения весов (вектор параметров модели),
    w_{new}  — новые значения весов после шага градиентного спуска,
    η (эта)  — скорость обучения (learning rate), малая положительная константа,
    ∇F(w)    — градиент функции потерь по весам.
Смысл:
    - градиент показывает, в какую сторону функция растёт сильнее всего
    - мы вычитаем η * ∇F, то есть делаем небольшой шаг в противоположную сторону, уменьшая значение функции потерь
    - повторяя это много раз, мы постепенно приближаемся к минимуму функции.


4. Что такое тензор во фреймворках?
Тензор — это обобщение скаляров, векторов и матриц.
В практическом смысле в TensorFlow/PyTorch тензор можно воспринимать как
многомерный массив чисел (например, 1D, 2D, 3D, 4D массив и выше).
Примеры:
    - скаляр (одно число) — тензор нулевого порядка
    - вектор (строка чисел) — тензор первого порядка
    - матрица (таблица чисел) — тензор второго порядка
    - «куб» чисел (например, изображение с каналами) — тензор третьего порядка.
Во фреймворках тензоры:
    - хранят данные и веса моделей
    - могут эффективно обрабатываться на CPU и GPU
    - поддерживают операции дифференцирования (автоматический подсчёт градиентов),
      что нужно для обучения нейросетей методом градиентного спуска.


5. По какой причине метод градиентного спуска может остановиться в задаче оптимизации функции потерь?
Основные причины:
    1)  Градиент почти равен нулю.
        Это означает, что мы находимся в точке минимума (локального или глобального),
        либо на плато/седловой точке, где изменение функции по параметрам слабое.
        В этом случае шаги обновления становятся очень маленькими, и веса практически перестают меняться.
    2)  Достигнут лимит по количеству итераций (эпох).
        Алгоритм останавливается, потому что мы сами задали ограничение на длительность обучения.
    3)  Слишком маленькая скорость обучения.
        Шаги получаются настолько малы, что дальнейшие изменения весов не видны
        на практике, и кажется, что алгоритм «застыл».
В корректно настроенном обучении основная ожидаемая причина остановки — градиент близок к нулю, 
то есть мы дошли до минимума функции потерь (или до области, где функция почти не меняется).


6. При нахождении хорошо аппроксимирующего полинома к какому значению стремится функция потерь?
В нашей работе в качестве функции потерь используется среднеквадратичная ошибка (MSE — mean squared error).
Если полином хорошо аппроксимирует данные, то разница между предсказанными
значениями и истинными ответами становится маленькой.
В идеальном случае (без шума) функция потерь стремится к нулю.
В реальной задаче (с шумом) значение функции потерь стремится к небольшому положительному числу,
близкому к нулю, которое соответствует «лучшей» возможной аппроксимации при заданном уровне шума и выбранной модели.
'''
