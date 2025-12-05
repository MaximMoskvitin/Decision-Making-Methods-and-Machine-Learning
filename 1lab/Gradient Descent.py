import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# НАСТРОЙКИ ОБУЧЕНИЯ
# ==========================

# Параметры обучения для квадратичной регрессии
QUAD_LEARNING_RATE = 0.01
QUAD_EPOCHS = 1000

# Параметры обучения для линейной регрессии
LIN_LEARNING_RATE = 0.01
LIN_EPOCHS = 1000

# Доля объектов, идущих в тестовую выборку
TEST_SIZE = 0.2
RANDOM_SEED = 42


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def mse_loss(y_true, y_pred):
    """
    Среднеквадратичная ошибка: MSE = mean((y_true - y_pred)^2)
    """
    return tf.reduce_mean((y_true - y_pred) ** 2)


def create_quadratic_dataset(m,
                             min_value,
                             max_value,
                             k1_true,
                             k0_true,
                             b_true,
                             noise_std=1.0,
                             seed=None):
    """
    Создание искусственных данных для квадратичной регрессии.
    y = k1 * x^2 + k0 * x + b + noise
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    x = tf.random.uniform(shape=(m,),
                          minval=min_value,
                          maxval=max_value,
                          dtype=tf.float32)

    noise = tf.random.normal(shape=(m,),
                             mean=0.0,
                             stddev=noise_std,
                             dtype=tf.float32)

    y = k1_true * x ** 2 + k0_true * x + b_true + noise
    return x.numpy(), noise.numpy(), y.numpy()


def create_linear_dataset(m,
                          min_value,
                          max_value,
                          k0_true,
                          b_true,
                          noise_std=1.0,
                          seed=None):
    """
    Создание искусственных данных для линейной регрессии.
    y = k0 * x + b + noise
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    x = tf.random.uniform(shape=(m,),
                          minval=min_value,
                          maxval=max_value,
                          dtype=tf.float32)

    noise = tf.random.normal(shape=(m,),
                             mean=0.0,
                             stddev=noise_std,
                             dtype=tf.float32)

    y = k0_true * x + b_true + noise
    return x.numpy(), noise.numpy(), y.numpy()


def train_test_split_numpy(x, y, test_size=0.2, seed=42):
    """
    Простое разбиение numpy-массивов x и y на обучающую и тестовую выборки.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    test_count = int(len(indices) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    return x_train, x_test, y_train, y_test


def inspect_dataset(x, noise, y, title):
    """
    Печать основных характеристик набора данных и базовый scatter-график.
    """
    print(title)
    print("Форма x:", x.shape)
    print("Форма шума noise:", noise.shape)
    print("Форма y:", y.shape)
    print("Пример первых 5 значений x:", x[:5])
    print("Пример первых 5 значений y:", y[:5], "\n")

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label="исходные данные", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


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


def plot_quadratic_fit(x_train, y_train,
                       x_test, y_test,
                       k1_true, k0_true, b_true,
                       k1_learned, k0_learned, b_learned):
    """
    Визуализация квадратичной регрессии:
      - точки обучающей и тестовой выборок
      - истинный полином и найденный полином.
    """
    plt.figure(figsize=(6, 4))

    # Строим полиномы по всей области значений
    x_all = np.linspace(x_train.min(), x_train.max(), 200)
    y_true_all = k1_true * x_all ** 2 + k0_true * x_all + b_true
    y_pred_all = k1_learned * x_all ** 2 + k0_learned * x_all + b_learned

    # Обучающие и тестовые точки
    plt.scatter(x_train, y_train, label="обучающие точки", alpha=0.7)
    plt.scatter(x_test, y_test, label="тестовые точки", alpha=0.7)

    # Линии истинного и найденного полиномов
    plt.plot(x_all, y_true_all, label="истинный полином", linewidth=2)
    plt.plot(x_all, y_pred_all, label="найденный полином", linewidth=2, linestyle="--")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Квадратичная регрессия: истинная и найденная модель")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_linear_fit(x_train, y_train,
                    x_test, y_test,
                    k0_true, b_true,
                    k0_learned, b_learned):
    """
    Визуализация линейной регрессии:
      - точки обучающей и тестовой выборок
      - истинная прямая и найденная прямая.
    """
    plt.figure(figsize=(6, 4))

    x_all = np.linspace(x_train.min(), x_train.max(), 200)
    y_true_all = k0_true * x_all + b_true
    y_pred_all = k0_learned * x_all + b_learned

    plt.scatter(x_train, y_train, label="обучающие точки", alpha=0.7)
    plt.scatter(x_test, y_test, label="тестовые точки", alpha=0.7)

    plt.plot(x_all, y_true_all, label="истинная прямая", linewidth=2)
    plt.plot(x_all, y_pred_all, label="найденная прямая", linewidth=2, linestyle="--")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Линейная регрессия: истинная и найденная модель")
    plt.grid(True)
    plt.legend()
    plt.show()


# ==========================
# ГРАДИЕНТНЫЙ СПУСК: ОБУЧЕНИЕ
# ==========================

def train_quadratic_regression(x_train, y_train,
                               learning_rate=QUAD_LEARNING_RATE,
                               epochs=QUAD_EPOCHS):
    """
    Обучение квадратичной регрессии методом градиентного спуска.
    Модель: y_hat = k1 * x^2 + k0 * x + b
    """
    x_train_tf = tf.constant(x_train, dtype=tf.float32)
    y_train_tf = tf.constant(y_train, dtype=tf.float32)

    # инициализация весов модели как тензоров tf.Variable
    k1 = tf.Variable(tf.random.normal(shape=()), name="k1")
    k0 = tf.Variable(tf.random.normal(shape=()), name="k0")
    b = tf.Variable(tf.random.normal(shape=()), name="b")

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = k1 * x_train_tf ** 2 + k0 * x_train_tf + b
            loss = mse_loss(y_train_tf, y_pred)

        grads = tape.gradient(loss, [k1, k0, b])
        optimizer.apply_gradients(zip(grads, [k1, k0, b]))

        loss_history.append(loss.numpy())

        # периодически выводим информацию об обучении
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[КВАДРАТИЧНАЯ] Epoch {epoch + 1:4d}/{epochs}: loss = {loss.numpy():.6f}")

    return k1.numpy(), k0.numpy(), b.numpy(), loss_history


def train_linear_regression(x_train, y_train,
                            learning_rate=LIN_LEARNING_RATE,
                            epochs=LIN_EPOCHS):
    """
    Обучение линейной регрессии методом градиентного спуска.
    Модель: y_hat = k0 * x + b
    """
    x_train_tf = tf.constant(x_train, dtype=tf.float32)
    y_train_tf = tf.constant(y_train, dtype=tf.float32)

    k0 = tf.Variable(tf.random.normal(shape=()), name="k0_linear")
    b = tf.Variable(tf.random.normal(shape=()), name="b_linear")

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = k0 * x_train_tf + b
            loss = mse_loss(y_train_tf, y_pred)

        grads = tape.gradient(loss, [k0, b])
        optimizer.apply_gradients(zip(grads, [k0, b]))

        loss_history.append(loss.numpy())

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[ЛИНЕЙНАЯ]     Epoch {epoch + 1:4d}/{epochs}: loss = {loss.numpy():.6f}")

    return k0.numpy(), b.numpy(), loss_history


# ==========================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================

def main():
    # фиксируем seed для воспроизводимости
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -------------
    # Часть 1. Квадратичная регрессия
    # -------------

    print("=== ЧАСТЬ 1. КВАДРАТИЧНАЯ РЕГРЕССИЯ ===\n")

    m_quad = 200
    min_value_quad = -3.0
    max_value_quad = 3.0

    k1_true = 0.7
    k0_true = -1.2
    b_true = 2.0
    noise_std_quad = 1.0

    x_quad, noise_quad, y_quad = create_quadratic_dataset(
        m=m_quad,
        min_value=min_value_quad,
        max_value=max_value_quad,
        k1_true=k1_true,
        k0_true=k0_true,
        b_true=b_true,
        noise_std=noise_std_quad,
        seed=RANDOM_SEED
    )

    inspect_dataset(x_quad, noise_quad, y_quad,
                    "Искусственные данные (квадратичная регрессия)")

    # разделяем данные на обучающую и тестовую выборки
    x_quad_train, x_quad_test, y_quad_train, y_quad_test = train_test_split_numpy(
        x_quad,
        y_quad,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED
    )

    print("Размер обучающей выборки (квадрат):", x_quad_train.shape[0])
    print("Размер тестовой выборки (квадрат):", x_quad_test.shape[0], "\n")

    # обучаем модель на обучающей выборке
    k1_learned, k0_learned, b_learned, loss_history_quad = train_quadratic_regression(
        x_quad_train,
        y_quad_train,
        learning_rate=QUAD_LEARNING_RATE,
        epochs=QUAD_EPOCHS
    )

    # оцениваем качество на train и test
    y_quad_train_pred = k1_learned * x_quad_train ** 2 + k0_learned * x_quad_train + b_learned
    y_quad_test_pred = k1_learned * x_quad_test ** 2 + k0_learned * x_quad_test + b_learned

    train_mse_quad = np.mean((y_quad_train - y_quad_train_pred) ** 2)
    test_mse_quad = np.mean((y_quad_test - y_quad_test_pred) ** 2)

    print(f"[КВАДРАТИЧНАЯ] Итоговые параметры:")
    print(f"  истинные:   k1 = {k1_true:.3f}, k0 = {k0_true:.3f}, b = {b_true:.3f}")
    print(f"  найденные:  k1 = {k1_learned:.3f}, k0 = {k0_learned:.3f}, b = {b_learned:.3f}")
    print(f"  MSE на train: {train_mse_quad:.4f}")
    print(f"  MSE на test:  {test_mse_quad:.4f}\n")

    plot_loss(loss_history_quad,
              title="Изменение функции потерь (квадратичная регрессия)")
    plot_quadratic_fit(
        x_quad_train, y_quad_train,
        x_quad_test, y_quad_test,
        k1_true, k0_true, b_true,
        k1_learned, k0_learned, b_learned
    )

    # -------------
    # Часть 2. Линейная регрессия
    # -------------

    print("\n=== ЧАСТЬ 2. ЛИНЕЙНАЯ РЕГРЕССИЯ ===\n")

    m_lin = 200
    min_value_lin = -3.0
    max_value_lin = 3.0

    k0_true_lin = 2.5
    b_true_lin = -1.0
    noise_std_lin = 1.0

    x_lin, noise_lin, y_lin = create_linear_dataset(
        m=m_lin,
        min_value=min_value_lin,
        max_value=max_value_lin,
        k0_true=k0_true_lin,
        b_true=b_true_lin,
        noise_std=noise_std_lin,
        seed=RANDOM_SEED
    )

    inspect_dataset(x_lin, noise_lin, y_lin,
                    "Искусственные данные (линейная регрессия)")

    x_lin_train, x_lin_test, y_lin_train, y_lin_test = train_test_split_numpy(
        x_lin,
        y_lin,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED
    )

    print("Размер обучающей выборки (линейная):", x_lin_train.shape[0])
    print("Размер тестовой выборки (линейная):", x_lin_test.shape[0], "\n")

    k0_learned_lin, b_learned_lin, loss_history_lin = train_linear_regression(
        x_lin_train,
        y_lin_train,
        learning_rate=LIN_LEARNING_RATE,
        epochs=LIN_EPOCHS
    )

    y_lin_train_pred = k0_learned_lin * x_lin_train + b_learned_lin
    y_lin_test_pred = k0_learned_lin * x_lin_test + b_learned_lin

    train_mse_lin = np.mean((y_lin_train - y_lin_train_pred) ** 2)
    test_mse_lin = np.mean((y_lin_test - y_lin_test_pred) ** 2)

    print(f"[ЛИНЕЙНАЯ]     Итоговые параметры:")
    print(f"  истинные:   k0 = {k0_true_lin:.3f}, b = {b_true_lin:.3f}")
    print(f"  найденные:  k0 = {k0_learned_lin:.3f}, b = {b_learned_lin:.3f}")
    print(f"  MSE на train: {train_mse_lin:.4f}")
    print(f"  MSE на test:  {test_mse_lin:.4f}\n")

    plot_loss(loss_history_lin,
              title="Изменение функции потерь (линейная регрессия)")
    plot_linear_fit(
        x_lin_train, y_lin_train,
        x_lin_test, y_lin_test,
        k0_true_lin, b_true_lin,
        k0_learned_lin, b_learned_lin
    )


if __name__ == "__main__":
    main()


'''
1. Что такое градиентный спуск и как он используется в машинном обучении?
Градиентный спуск — это итерационный метод поиска минимума функции.
В машинном обучении этой функцией обычно является функция потерь,
которая измеряет, насколько сильно предсказания модели отличаются от правильных ответов.
Идея метода:
    1) считаем значение функции потерь на текущих весах модели;
    2) вычисляем градиент (направление наибольшего роста функции);
    3) делаем шаг в противоположную сторону (по антиградиенту), чтобы уменьшить значение функции потерь;
    4) повторяем шаги до тех пор, пока ошибка не станет достаточно маленькой
       или пока не закончится количество эпох обучения.
Так модель постепенно «учится» и подбирает такие веса, при которых
функция потерь минимальна или близка к минимуму.


2. Что такое градиент в контексте градиентного спуска?
Градиент — это вектор частных производных функции по всем её параметрам.
В контексте градиентного спуска:
    - мы рассматриваем функцию потерь L(w), где w — вектор весов модели;
    - градиент ∇L(w) показывает направление наискорейшего роста функции потерь;
    - отрицательный градиент −∇L(w) показывает направление наискорейшего убывания.
По сути, градиент говорит, как нужно изменять каждый параметр модели,
чтобы увеличить или уменьшить значение функции потерь.
В градиентном спуске мы двигаемся по направлению минус градиента,
чтобы L(w) становилась всё меньше.


3. Как обновляются веса в градиентном спуске?
Обновление весов происходит по формуле:
    w_{new} = w_{old} - η * ∇L(w_{old}),
где
    w — вектор весов (параметров) модели,
    η (eta) — скорость обучения (learning rate),
    ∇L(w_{old}) — градиент функции потерь по весам на текущем шаге.
Смысл:
    - считаем, как изменение каждого веса влияет на ошибку;
    - умножаем градиент на шаг обучения η (чтобы не делать слишком резких шагов);
    - вычитаем этот вклад из текущих весов.
В фреймворках (как TensorFlow в этой работе) это делается автоматически:
оптимизатор (SGD) получает градиенты и обновляет переменные tf.Variable.


4. Что такое тензор во фреймворках?
Тензор — это обобщение понятия скаляра, вектора и матрицы.
Можно думать так:
    - скаляр — тензор нулевого ранга (одно число);
    - вектор — тензор первого ранга (одномерный массив);
    - матрица — тензор второго ранга (двумерный массив);
    - тензор более высокого ранга — многомерный массив.
Во фреймворках (TensorFlow, PyTorch) тензор — это основной тип данных для хранения
и обработки чисел на CPU/GPU. Он:
    - хранит значения (например, признаки, веса, градиенты);
    - имеет форму (shape), тип данных (dtype) и устройство (device);
    - может участвовать в автоматическом дифференцировании
      (через tf.GradientTape, autograd и т.п.).


5. По какой причине метод градиентного спуска может остановиться в задаче оптимизации функции потерь?
Есть несколько типичных причин:
    1) Мы дошли до минимума (или окрестности минимума):
       - градиент по всем параметрам близок к нулю;
       - шаги обновления становятся очень маленькими;
       - значение функции потерь почти не меняется.
    2) Достигнуто максимальное число эпох / итераций:
       - мы явно ограничили количество шагов обучения.
    3) Слишком маленькая скорость обучения:
       - шаги настолько малы, что уменьшение функции потерь почти не заметно;
       - на практике кажется, что алгоритм «застыл».
    4) Слишком большая скорость обучения:
       - значения весов скачут, функция потерь может не уменьшаться
         или даже расти, и обучение прерывают вручную.
В корректно настроенном обучении главная ожидаемая причина остановки —
градиент становится близок к нулю, то есть мы находимся в точке минимума
или на плато, где функция потерь почти не меняется.


6. При нахождении хорошо аппроксимирующего полинома к какому значению стремится функция потерь?
В нашей работе в качестве функции потерь используется среднеквадратичная ошибка (MSE — mean squared error).
Если полином хорошо аппроксимирует данные, то разница между предсказанными
значениями и истинными ответами становится маленькой.
    - В идеальном случае (без шума) функция потерь стремится к нулю.
    - В реальной задаче (с шумом) MSE стремится к небольшому положительному числу,
      близкому к нулю, которое соответствует «лучшей» возможной аппроксимации
      при заданном уровне шума и выбранной модели.
'''
