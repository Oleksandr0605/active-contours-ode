import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import inv
from scipy.interpolate import RectBivariateSpline

class ExternalForce:
    def __init__(self, image, sigma=5.0):
        """
        Обчислює поле сил із зображення.
        sigma: сила розмиття (чим більше, тим далі змія 'бачить' край)
        """
        self.H, self.W = image.shape
        
        # 1. Розмиття (щоб розширити басейн притягання)
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # 2. Градієнти зображення (Sobel)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        
        # 3. Магнітуда градієнта (енергія країв)
        # Ми хочемо максимізувати це значення на контурі -> потенціал E = -magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Нормалізація для зручності (0..1)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # 4. Сили - це градієнт від потенціалу енергії
        # Оскільки ми хочемо притягнутися до максимуму магнітуди, сила = градієнт(магнітуди)
        self.fx = cv2.Sobel(magnitude, cv2.CV_64F, 1, 0, ksize=3)
        self.fy = cv2.Sobel(magnitude, cv2.CV_64F, 0, 1, ksize=3)
        
        # Нормалізуємо сили, щоб змія не "вилетіла"
        force_mag = np.sqrt(self.fx**2 + self.fy**2)
        self.fx /= (force_mag + 1e-5)
        self.fy /= (force_mag + 1e-5)
        
        # 5. Інтерполятори (щоб брати силу в дробових координатах)
        x_range = np.arange(self.W)
        y_range = np.arange(self.H)
        # Увага: RectBivariateSpline приймає (y, x)
        self.interp_fx = RectBivariateSpline(y_range, x_range, self.fx)
        self.interp_fy = RectBivariateSpline(y_range, x_range, self.fy)

    def get_force(self, points):
        """Повертає вектори сил (Fx, Fy) для заданих координат точок"""
        # points: (N, 2) -> (x, y)
        # interp приймає (y, x) і grid=False для поточкового запиту
        fx_vals = self.interp_fx(points[:, 1], points[:, 0], grid=False)
        fy_vals = self.interp_fy(points[:, 1], points[:, 0], grid=False)
        return np.column_stack([fx_vals, fy_vals])

class Snake:
    def __init__(self, start_points, alpha=0.01, beta=0.1, gamma=1.0, w_line=10.0):
        """
        w_line: вага зовнішньої сили (gamma * w_line)
        """
        self.points = start_points
        self.n_points = len(start_points)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w_line = w_line
        
        self.A_inv = self._create_internal_matrix()

    def _create_internal_matrix(self):
        N = self.n_points
        alpha, beta = self.alpha, self.beta
        a, b, c = beta, -alpha - 4*beta, 2*alpha + 6*beta
        
        A = np.zeros((N, N))
        for i in range(N):
            A[i, (i-2)%N] = a
            A[i, (i-1)%N] = b
            A[i, i]       = c
            A[i, (i+1)%N] = b
            A[i, (i+2)%N] = a
            
        return inv(np.eye(N) + self.gamma * A)

    def step(self, external_force_field):
        # 1. Отримуємо зовнішні сили в поточних точках
        f_ext = external_force_field.get_force(self.points)
        
        # 2. Рівняння руху: x = Inv(A) * (x_old + gamma * w * F_ext)
        # Обмежуємо силу (kappa), щоб не було вибухів
        total_force = self.gamma * self.w_line * f_ext
        
        self.points = self.A_inv @ (self.points + total_force)
        
        # Обмеження, щоб точки не вилетіли за межі картинки
        H, W = external_force_field.H, external_force_field.W
        self.points[:, 0] = np.clip(self.points[:, 0], 0, W-1)
        self.points[:, 1] = np.clip(self.points[:, 1], 0, H-1)

    def get_points(self):
        return np.vstack([self.points, self.points[0]])

# --- Генерація синтетичних даних і запуск ---

def create_synthetic_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    # Малюємо біле коло (об'єкт)
    cv2.circle(img, (100, 100), 40, 255, -1)
    # Малюємо білий прямокутник (ще об'єкт)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    return img

if __name__ == "__main__":
    # 1. Дані
    img = create_synthetic_image()
    
    # 2. Поле сил
    ext_force = ExternalForce(img, sigma=3.0)
    
    # 3. Ініціалізація змії (велике коло навколо центру)
    s = np.linspace(0, 2*np.pi, 100, endpoint=False)
    # Коло радіусом 90 навколо (100, 100)
    x = 100 + 90 * np.cos(s)
    y = 100 + 90 * np.sin(s)
    init_points = np.column_stack([x, y])
    
    # Параметри: w_line - це "жадібність" змії до країв
    snake = Snake(init_points, alpha=0.5, beta=1.0, gamma=1.0, w_line=5.0)    
    
    # Візуалізація
    plt.figure(figsize=(10, 5))
    
    # Показуємо поле сил (проріджене для краси)
    plt.subplot(1, 2, 1)
    plt.title("Field of external forces (Fx, Fy)")
    plt.imshow(img, cmap='gray', alpha=0.3)
    X, Y = np.meshgrid(np.arange(0, 200, 10), np.arange(0, 200, 10))
    # Вектори сил на сітці
    FX = ext_force.interp_fx(Y[:,0], X[0,:]) # Увага на порядок осей
    FY = ext_force.interp_fy(Y[:,0], X[0,:])
    plt.quiver(X, Y, FX, FY, color='red', scale=30)
    
    # Анімація змії
    plt.subplot(1, 2, 2)
    plt.title("Evolution Snake")
    plt.imshow(img, cmap='gray')
    
    snake_line, = plt.plot([], [], 'g-', lw=2)
    
    # Запускаємо 100 кроків
    for t in range(800):
        snake.step(ext_force)
        
        if t % 2 == 0:  # Оновлюємо графік кожні 2 кроки
            pts = snake.get_points()
            snake_line.set_data(pts[:, 0], pts[:, 1])
            plt.pause(0.01)
            
    plt.show()