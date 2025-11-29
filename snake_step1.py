import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

class Snake:
    def __init__(self, n_points=100, center=(0.5, 0.5), radius=0.3, alpha=0.01, beta=0.1, gamma=0.1):
        """
        Ініціалізація змії.
        alpha: вага першої похідної (еластичність, стягування)
        beta: вага другої похідної (жорсткість, гладкість)
        gamma: крок часу (time step)
        """
        self.n_points = n_points
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 1. Створюємо початковий контур (коло)
        s = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = center[0] + radius * np.cos(s)
        y = center[1] + radius * np.sin(s)
        
        # Зберігаємо координати як матрицю (N, 2)
        self.points = np.column_stack([x, y])
        
        # 2. Обрахунок матриці внутрішніх сил (один раз!)
        self.A_inv = self._create_internal_matrix()

    def _create_internal_matrix(self):
        """
        Створює пентадіагональну матрицю для напівнеявного методу.
        Рівняння: (I + gamma * A) * x_{t+1} = x_t
        Нам потрібна обернена матриця: M = (I + gamma * A)^(-1)
        """
        N = self.n_points
        alpha = self.alpha
        beta = self.beta
        
        # Коефіцієнти
        a = beta
        b = -alpha - 4*beta
        c = 2*alpha + 6*beta
        
        # Створюємо матрицю A (циклічну)
        A = np.zeros((N, N))
        for i in range(N):
            A[i, (i-2)%N] = a
            A[i, (i-1)%N] = b
            A[i, i]       = c
            A[i, (i+1)%N] = b
            A[i, (i+2)%N] = a
            
        # Додаємо одиничну матрицю (I) і множимо на gamma
        # (I + gamma * A)
        B = np.eye(N) + self.gamma * A
        
        # Повертаємо обернену
        return inv(B)

    def add_noise(self, noise_level=0.01):
        """Додаємо випадковий шум до координат, щоб перевірити згладжування"""
        noise = np.random.normal(0, noise_level, self.points.shape)
        self.points += noise

    def step(self):
        """
        Один крок симуляції.
        x_{t+1} = M @ (x_t + gamma * F_ext)
        Поки що F_ext = 0.
        """
        # Множення матриці на координати (окремо для X та Y, або разом)
        # points має форму (N, 2), A_inv (N, N). Результат (N, 2)
        self.points = self.A_inv @ self.points

    def get_points(self):
        # Замикаємо контур для красивої візуалізації (додаємо першу точку в кінець)
        return np.vstack([self.points, self.points[0]])

# --- Запуск симуляції ---

if __name__ == "__main__":
    # Параметри
    snake = Snake(n_points=100, alpha=0.05, beta=0.5, gamma=1.0)
    
    # Зіпсуємо ідеальне коло шумом
    snake.add_noise(noise_level=0.05)
    
    plt.figure(figsize=(8, 8))
    plt.title("Еволюція активного контуру (тільки внутрішні сили)")
    
    # Малюємо початковий стан
    pts = snake.get_points()
    plt.plot(pts[:, 0], pts[:, 1], 'r--', label='Початковий (шум)')
    
    # Робимо кроки
    for t in range(1, 51):
        snake.step()
        
        # Малюємо проміжні стани
        if t % 10 == 0:
            pts = snake.get_points()
            alpha_val = t / 50.0  # Прозорість
            plt.plot(pts[:, 0], pts[:, 1], 'b', alpha=alpha_val, label=f'Крок {t}')

    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()