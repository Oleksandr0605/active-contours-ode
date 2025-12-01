import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import inv
from scipy.interpolate import RectBivariateSpline

# --- 1. Класи з вашого коду (Snake Step 2) ---

class ExternalForce:
    def __init__(self, image, sigma=5.0):
        self.H, self.W = image.shape
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Градієнти (Sobel)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Потенціал P = -magnitude
        self.potential = -magnitude 

        # Нормалізація для градієнта сил
        magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        self.fx = cv2.Sobel(magnitude_norm, cv2.CV_64F, 1, 0, ksize=3)
        self.fy = cv2.Sobel(magnitude_norm, cv2.CV_64F, 0, 1, ksize=3)
        
        force_mag = np.sqrt(self.fx**2 + self.fy**2)
        self.fx /= (force_mag + 1e-5)
        self.fy /= (force_mag + 1e-5)
        
        x_range = np.arange(self.W)
        y_range = np.arange(self.H)
        self.interp_fx = RectBivariateSpline(y_range, x_range, self.fx)
        self.interp_fy = RectBivariateSpline(y_range, x_range, self.fy)
        self.interp_pot = RectBivariateSpline(y_range, x_range, self.potential)

    def get_force(self, points):
        fx_vals = self.interp_fx(points[:, 1], points[:, 0], grid=False)
        fy_vals = self.interp_fy(points[:, 1], points[:, 0], grid=False)
        return np.column_stack([fx_vals, fy_vals])

    def get_potential(self, points):
        potential_vals = self.interp_pot(points[:, 1], points[:, 0], grid=False)
        return np.sum(potential_vals)

class Snake:
    def __init__(self, start_points, alpha=0.01, beta=0.1, gamma=1.0, w_line=10.0):
        self.points = start_points.copy()
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
        f_ext = external_force_field.get_force(self.points)
        total_force = self.gamma * self.w_line * f_ext
        self.points = self.A_inv @ (self.points + total_force)
        H, W = external_force_field.H, external_force_field.W
        self.points[:, 0] = np.clip(self.points[:, 0], 0, W-1)
        self.points[:, 1] = np.clip(self.points[:, 1], 0, H-1)
    
    def _get_derivatives(self):
        points = self.points
        v_prime = points - np.roll(points, 1, axis=0)
        v_second_prime = np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
        return v_prime, v_second_prime

    def get_internal_energy(self):
        v_prime, v_second_prime = self._get_derivatives()
        norm_v_prime_sq = np.sum(v_prime**2, axis=1)
        norm_v_second_prime_sq = np.sum(v_second_prime**2, axis=1)
        E_int_sum = self.alpha * norm_v_prime_sq + self.beta * norm_v_second_prime_sq
        return 0.5 * np.sum(E_int_sum)
        
    def get_total_energy(self, external_force_field):
        E_int = self.get_internal_energy()
        E_ext = self.w_line * external_force_field.get_potential(self.points) 
        return E_int + E_ext

# --- 2. Генерація даних ---
def create_synthetic_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, 255, -1) # Об'єкт
    return img

def create_optimal_snake():
    s = np.linspace(0, 2*np.pi, 100, endpoint=False)
    x = 100 + 80 * np.cos(s)
    y = 100 + 80 * np.sin(s)
    return np.column_stack([x, y])

# --- 3. Основний код для генерації графіка (Step 5.1) ---
if __name__ == "__main__":
    # Налаштування
    img = create_synthetic_image()
    ext_force = ExternalForce(img, sigma=3.0)
    
    # Ініціалізація
    init_points = create_optimal_snake()
    snake = Snake(init_points, alpha=0.5, beta=1.0, gamma=1.0, w_line=5.0)
    
    # Збір даних
    steps = 300
    energy_history = []
    
    print("Запуск симуляції для аналізу енергії...")
    for t in range(steps):
        # 1. Записуємо поточну енергію
        current_energy = snake.get_total_energy(ext_force)
        energy_history.append(current_energy)
        
        # 2. Робимо крок
        snake.step(ext_force)
        
    # --- 4. Побудова графіка ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(energy_history, color='blue', linewidth=2, label=r'$E_{total}(t)$')
    
    # Додамо лінію тренду або асимптоту (останнє значення)
    final_energy = energy_history[-1]
    plt.axhline(y=final_energy, color='red', linestyle='--', alpha=0.7, label=f'Min Energy: {final_energy:.1f}')
    
    plt.title('Convergence Analysis: Evolution of Total Energy', fontsize=14)
    plt.xlabel('Iteration Step ($t$)', fontsize=12)
    plt.ylabel('Total Energy ($E_{int} + E_{ext}$)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Текстова вставка на графіку
    plt.text(steps * 0.5, (energy_history[0] + final_energy)/2, 
             "Monotonic Decay $\\rightarrow$ Stability", 
             fontsize=12, color='green', fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8))

    output_file = 'energy_convergence_plot.png'
    plt.savefig(output_file, dpi=300)
    print(f"Графік успішно збережено у файл: {output_file}")
    plt.show()