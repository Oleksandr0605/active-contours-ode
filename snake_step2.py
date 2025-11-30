import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import inv
from scipy.interpolate import RectBivariateSpline
import os

# --- Клас ExternalForce: Обчислення зовнішнього поля та потенціалу (Step 2 & 4) ---
class ExternalForce:
    def __init__(self, image, sigma=5.0):
        """
        Обчислює поле сил (градієнт) та потенціал (-магнітуда градієнта).
        """
        self.H, self.W = image.shape
        
        # 1. Розмиття (щоб розширити басейн притягання)
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # 2. Градієнти зображення (Sobel)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        
        # 3. Магнітуда градієнта (енергія країв)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Створюємо потенціал P = -magnitude. Це E_ext для однієї точки.
        self.potential = -magnitude 

        # Нормалізація для зручності (0..1)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # 4. Сили - це градієнт від магнітуди
        self.fx = cv2.Sobel(magnitude, cv2.CV_64F, 1, 0, ksize=3)
        self.fy = cv2.Sobel(magnitude, cv2.CV_64F, 0, 1, ksize=3)
        
        # Нормалізуємо сили
        force_mag = np.sqrt(self.fx**2 + self.fy**2)
        self.fx /= (force_mag + 1e-5)
        self.fy /= (force_mag + 1e-5)
        
        # 5. Інтерполятори
        x_range = np.arange(self.W)
        y_range = np.arange(self.H)
        self.interp_fx = RectBivariateSpline(y_range, x_range, self.fx)
        self.interp_fy = RectBivariateSpline(y_range, x_range, self.fy)
        self.interp_pot = RectBivariateSpline(y_range, x_range, self.potential)

    def get_force(self, points):
        """Повертає вектори сил (Fx, Fy) для заданих координат точок"""
        fx_vals = self.interp_fx(points[:, 1], points[:, 0], grid=False)
        fy_vals = self.interp_fy(points[:, 1], points[:, 0], grid=False)
        return np.column_stack([fx_vals, fy_vals])

    def get_potential(self, points):
        """Повертає суму потенційної енергії E_ext для заданих координат точок"""
        potential_vals = self.interp_pot(points[:, 1], points[:, 0], grid=False)
        return np.sum(potential_vals)

# --- Клас Snake: Динаміка, внутрішні сили та енергія (Step 1 & 4) ---
class Snake:
    def __init__(self, start_points, alpha=0.01, beta=0.1, gamma=1.0, w_line=10.0):
        self.points = start_points.copy() # Копія, щоб не змінювати init_points
        self.n_points = len(start_points)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w_line = w_line
        
        self.A_inv = self._create_internal_matrix()

    def _create_internal_matrix(self):
        N = self.n_points
        alpha, beta = self.alpha, self.beta
        # Коефіцієнти для пентадіагональної матриці A (дискретизація v'' та v''')
        a, b, c = beta, -alpha - 4*beta, 2*alpha + 6*beta
        
        A = np.zeros((N, N))
        for i in range(N):
            A[i, (i-2)%N] = a
            A[i, (i-1)%N] = b
            A[i, i]       = c
            A[i, (i+1)%N] = b
            A[i, (i+2)%N] = a
            
        # M = (I + gamma * A)^(-1)
        return inv(np.eye(N) + self.gamma * A)

    def step(self, external_force_field):
        """Один крок еволюції контуру"""
        f_ext = external_force_field.get_force(self.points)
        total_force = self.gamma * self.w_line * f_ext
        
        # Напівнеявна схема інтегрування: x_{t+1} = M * (x_t + gamma * F_ext)
        self.points = self.A_inv @ (self.points + total_force)
        
        # Обмеження координат
        H, W = external_force_field.H, external_force_field.W
        self.points[:, 0] = np.clip(self.points[:, 0], 0, W-1)
        self.points[:, 1] = np.clip(self.points[:, 1], 0, H-1)

    def get_points(self):
        return np.vstack([self.points, self.points[0]])
    
    # --- МЕТОДИ ДЛЯ АНАЛІЗУ ЕНЕРГІЇ (STEP 4) ---
    def _get_derivatives(self):
        """Обчислює дискретні похідні v' та v'' для формули енергії"""
        points = self.points
        
        # v' ≈ v_i - v_{i-1} (Еластичність)
        v_prime = points - np.roll(points, 1, axis=0)
        
        # v'' ≈ v_{i+1} - 2v_i + v_{i-1} (Жорсткість)
        v_second_prime = np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
        
        return v_prime, v_second_prime

    def get_internal_energy(self): # E_int
        """
        Обчислює внутрішню енергію. 
        E_{int} = 0.5 * SUM(alpha * |v_i'|^2 + beta * |v_i''|^2)
        """
        v_prime, v_second_prime = self._get_derivatives()
        
        norm_v_prime_sq = np.sum(v_prime**2, axis=1)
        norm_v_second_prime_sq = np.sum(v_second_prime**2, axis=1)
        
        E_int_sum = self.alpha * norm_v_prime_sq + self.beta * norm_v_second_prime_sq
        
        E_int = 0.5 * np.sum(E_int_sum)
        return E_int
        
    def get_total_energy(self, external_force_field): # E_total
        """Обчислює повну енергію E_total = E_int + E_ext"""
        E_int = self.get_internal_energy()
        # E_ext = w_line * SUM(P(v_i))
        E_ext = self.w_line * external_force_field.get_potential(self.points) 
        
        return E_int + E_ext


# --- Генерація синтетичних даних ---
def create_synthetic_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    # Малюємо біле коло (об'єкт)
    cv2.circle(img, (100, 100), 40, 255, -1)
    # Малюємо білий прямокутник (ще об'єкт)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    return img

def create_optimal_snake():
    """Створює початковий контур (велике коло, оптимальне для захоплення)"""
    s = np.linspace(0, 2*np.pi, 100, endpoint=False)
    # Коло радіусом 90 навколо (100, 100)
    x = 100 + 90 * np.cos(s)
    y = 100 + 90 * np.sin(s)
    return np.column_stack([x, y])

def create_distant_snake(): # <--- НОВА ФУНКЦІЯ ДЛЯ STEP 6
    """Створює початковий контур (мале коло далеко від об'єкта, для демонстрації збою)"""
    s = np.linspace(0, 2*np.pi, 100, endpoint=False)
    # Мале коло радіусом 5 навколо (10, 10) - далеко від градієнтів об'єкта
    x = 10 + 5 * np.cos(s)
    y = 10 + 5 * np.sin(s)
    return np.column_stack([x, y])

# --- Функція для Step 5: Дослідження параметрів ---
def run_parameter_study(alpha, beta, w_line, steps, title_suffix, output_dir, img, ext_force):
    """Виконує повний цикл симуляції для заданих параметрів і зберігає фінальний контур."""
    
    init_points = create_optimal_snake() # Використовуємо оптимальне ініціалізаційне коло
    snake = Snake(init_points, alpha=alpha, beta=beta, gamma=1.0, w_line=w_line)    
    
    # Виконуємо кроки
    energy_history = []
    for t in range(steps):
        snake.step(ext_force)
        energy_history.append(snake.get_total_energy(ext_force))
        
    # --- Візуалізація та Збереження Фінального Контуру ---
    
    plt.figure(figsize=(10, 5))
    
    # SUBPLOT 1: Поле сил 
    plt.subplot(1, 2, 1)
    plt.title(f"Field of forces ($\sigma=3.0$ / $w_{{ext}}={w_line}$)")
    plt.imshow(img, cmap='gray', alpha=0.3)
    X, Y = np.meshgrid(np.arange(0, 200, 10), np.arange(0, 200, 10))
    FX = ext_force.interp_fx(Y[:,0], X[0,:])
    FY = ext_force.interp_fy(Y[:,0], X[0,:])
    plt.quiver(X, Y, FX, FY, color='red', scale=30)
    
    # SUBPLOT 2: Фінальний контур
    plt.subplot(1, 2, 2)
    plt.title(f"Фінальний контур ({title_suffix})")
    plt.imshow(img, cmap='gray')
    
    pts = snake.get_points()
    plt.plot(pts[:, 0], pts[:, 1], 'g-', lw=2)
    plt.plot(pts[0, 0], pts[0, 1], 'ro', markersize=5, label='Початок')
    
    plt.tight_layout()
    
    filename = f'final_contour_{title_suffix.replace(", ", "_").replace("$", "").replace("=", "")}.png'
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()
    
    return snake.get_total_energy(ext_force), energy_history 

# --- Функція для Step 6: Аналіз Басейну Притягання ---
def run_capture_range_study(w_line, steps, output_dir, img, ext_force):
    """Демонструє обмеженість басейну притягання (Capture Range)."""
    
    # Використовуємо High Alpha, щоб при відсутності F_ext змія швидко стягнулася
    ALPHA = 1.0
    BETA = 0.01
    
    # 1. Ініціалізація далеко від об'єкта
    init_points = create_distant_snake() # <--- ВИКОРИСТАННЯ ДАЛЕКОГО КОНТУРУ
    
    # Використовуємо High Alpha, щоб вона гарантовано стиснулася
    snake = Snake(init_points, alpha=ALPHA, beta=BETA, gamma=1.0, w_line=w_line)    
    
    # Виконуємо кроки
    for t in range(steps):
        snake.step(ext_force)
        
    # --- Візуалізація та Збереження Фінального Контуру ---
    
    plt.figure(figsize=(10, 5))
    
    # SUBPLOT 1: Поле сил 
    plt.subplot(1, 2, 1)
    plt.title(f"Field of forces ($\sigma=3.0$ / $w_{{ext}}={w_line}$)")
    plt.imshow(img, cmap='gray', alpha=0.3)
    X, Y = np.meshgrid(np.arange(0, 200, 10), np.arange(0, 200, 10))
    FX = ext_force.interp_fx(Y[:,0], X[0,:])
    FY = ext_force.interp_fy(Y[:,0], X[0,:])
    plt.quiver(X, Y, FX, FY, color='red', scale=30)
    
    # SUBPLOT 2: Фінальний контур
    plt.subplot(1, 2, 2)
    plt.title("Фінальний контур (Збій Capture Range)")
    plt.imshow(img, cmap='gray')
    
    pts = snake.get_points()
    
    # Малюємо початковий контур (для порівняння)
    init_pts = create_distant_snake()
    plt.plot(init_pts[:, 0], init_pts[:, 1], 'r--', lw=1, label='Початковий контур')
    
    # Фінальний контур (має стиснутися в точку)
    plt.plot(pts[:, 0], pts[:, 1], 'g-', lw=2, label='Фінальний контур')
    plt.plot(pts[0, 0], pts[0, 1], 'ro', markersize=5) 
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    filename = 'final_contour_CaptureRangeFailure.png'
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()
    
    print(f"Збережено результат Capture Range: {output_dir}/{filename}")

# --- Функція для Step 4: Основний прогін з анімацією та енергією ---
def run_main_analysis(img, ext_force, output_dir_frames):
    
    init_points = create_optimal_snake() # Використовуємо оптимальне ініціалізаційне коло
    # Збалансовані параметри для демонстрації схожості та стабільності
    snake = Snake(init_points, alpha=0.5, beta=1.0, gamma=1.0, w_line=5.0)    
    
    energy_history = []
    
    # Візуалізація
    plt.figure(figsize=(10, 5))
    
    # SUBPLOT 1: Поле сил (один раз)
    plt.subplot(1, 2, 1)
    plt.title("Field of external forces (Fx, Fy)")
    plt.imshow(img, cmap='gray', alpha=0.3)
    X, Y = np.meshgrid(np.arange(0, 200, 10), np.arange(0, 200, 10))
    FX = ext_force.interp_fx(Y[:,0], X[0,:])
    FY = ext_force.interp_fy(Y[:,0], X[0,:])
    plt.quiver(X, Y, FX, FY, color='red', scale=30)
    
    # Підготовка SUBPLOT 2 для кадрів
    plt.subplot(1, 2, 2)
    plt.title("Evolution Snake (Frames)")
    
    # --- ЗАПУСК ТА ЗБЕРЕЖЕННЯ КАДРІВ (ЗАМІСТЬ АНІМАЦІЇ) ---
    
    for t in range(800):
        snake.step(ext_force)
        energy = snake.get_total_energy(ext_force)
        energy_history.append(energy)

        # Зберігаємо кадр кожні 40 кроків
        if t % 40 == 0: 
            pts = snake.get_points()
            
            # Перемальовуємо, щоб мати чистий кадр
            plt.subplot(1, 2, 2)
            plt.cla() 
            plt.imshow(img, cmap='gray')
            plt.plot(pts[:, 0], pts[:, 1], 'g-', lw=2)
            plt.title(f"Evolution Snake (Step {t})")
            plt.xlim(0, 200)
            plt.ylim(0, 200)
            
            # Зберігаємо кадр
            plt.gcf().set_size_inches(10, 5) 
            plt.savefig(f'{output_dir_frames}/snake_frame_{t:04d}.png')
            
    plt.close()
    return energy_history

# --- ОСНОВНИЙ ВИКОНАВЧИЙ БЛОК ---
if __name__ == "__main__":
    
    # --- 0. ПІДГОТОВКА СЕРЕДОВИЩА ---
    output_dir_frames = 'frames'
    output_dir_results = 'results_analysis'
    output_dir_params = 'results_params'
    output_dir_capture = 'results_capture' # <--- НОВА ПАПКА
    
    os.makedirs(output_dir_frames, exist_ok=True)
    os.makedirs(output_dir_results, exist_ok=True)
    os.makedirs(output_dir_params, exist_ok=True)
    os.makedirs(output_dir_capture, exist_ok=True) # <--- СТВОРЕННЯ НОВОЇ ПАПКИ
    
    print("--- Підготовка даних ---")
    img = create_synthetic_image()
    ext_force = ExternalForce(img, sigma=3.0)
    
    # =========================================================================
    # 1. RUN MAIN ANALYSIS (STEP 4) - АНАЛІЗ ЕНЕРГІЇ
    # =========================================================================
    print("\n--- 1. Запуск Основного Аналізу та збору Енергії (Step 4) ---")
    energy_history = run_main_analysis(img, ext_force, output_dir_frames)

    # 1.1. Графік Енергії (Step 4)
    if energy_history:
        plt.figure(figsize=(8, 4))
        plt.plot(energy_history, 'b-')
        plt.title("Еволюція Повної Енергії $E_{total}(t)$ (Функція Ляпунова)")
        plt.xlabel("Крок ітерації (t)")
        plt.ylabel("Енергія")
        plt.grid(True)
        energy_file = f'{output_dir_results}/snake_energy_evolution.png'
        plt.savefig(energy_file)
        plt.close()
        print(f"Графік енергії збережено: {energy_file}")
        # 
    
    # =========================================================================
    # 2. RUN PARAMETER STUDY (STEP 5) - ДОСЛІДЖЕННЯ ПАРАМЕТРІВ
    # =========================================================================
    print("\n--- 2. Дослідження Параметрів (Step 5) ---")
    
    STEPS = 400
    W_LINE = 5.0

    # 2.1. СЦЕНАРІЙ А: High α, Low β (ГУМКА)
    E_A, _ = run_parameter_study(
        alpha=1.0, 
        beta=0.01, 
        w_line=W_LINE, 
        steps=STEPS, 
        title_suffix="A: $\\alpha$=1.0, $\\beta$=0.01 (Гума)",
        output_dir=output_dir_params, 
        img=img, 
        ext_force=ext_force
    )

    # 2.2. СЦЕНАРІЙ B: Low α, High β (МЕТАЛЕВИЙ ДРІТ)
    E_B, _ = run_parameter_study(
        alpha=0.01, 
        beta=1.0, 
        w_line=W_LINE, 
        steps=STEPS, 
        title_suffix="B: $\\alpha$=0.01, $\\beta$=1.0 (Дріт)",
        output_dir=output_dir_params, 
        img=img, 
        ext_force=ext_force
    )
    
    # 2.3. СЦЕНАРІЙ C: Збалансований (Оптимальний)
    E_C, _ = run_parameter_study(
        alpha=0.5, 
        beta=1.0, 
        w_line=W_LINE, 
        steps=STEPS, 
        title_suffix="C: $\\alpha$=0.5, $\\beta$=1.0 (Збалансований)",
        output_dir=output_dir_params, 
        img=img, 
        ext_force=ext_force
    )

    print("\n--- Висновки Параметрів (див. папку results_params): ---")
    print(f"1. Сценарій А (Гума, High Alpha): Фінальна Енергія = {E_A:.2f}")
    print(f"2. Сценарій В (Дріт, High Beta): Фінальна Енергія = {E_B:.2f}")
    print(f"3. Сценарій С (Баланс): Фінальна Енергія = {E_C:.2f}")
    
    # =========================================================================
    # 3. RUN CAPTURE RANGE STUDY (STEP 6) - БАСЕЙН ПРИТЯГАННЯ
    # =========================================================================
    print("\n--- 3. Аналіз Басейну Притягання (Step 6) ---")
    run_capture_range_study(
        w_line=W_LINE, 
        steps=STEPS, 
        output_dir=output_dir_capture, 
        img=img, 
        ext_force=ext_force
    )

    print("\nУспішно завершено повний аналіз динамічної системи активного контуру. Дивіться результати у відповідних папках.")