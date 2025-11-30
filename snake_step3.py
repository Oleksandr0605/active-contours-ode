import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, eigvals

def analyze_snake_operator(n_points=100, alpha=0.01, beta=0.1, gamma=1.0):
    """
    Аналізує спектральні властивості матриці переходу змії.
    """
    N = n_points
    
    # --- 1. Відтворюємо побудову матриці A (як у вашому snake_step1.py) ---
    a = beta
    b = -alpha - 4*beta
    c = 2*alpha + 6*beta
    
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i-2)%N] = a
        A[i, (i-1)%N] = b
        A[i, i]       = c
        A[i, (i+1)%N] = b
        A[i, (i+2)%N] = a
        
    # --- 2. Оператор еволюції M ---
    # Рівняння: (I + gamma * A) * x_{t+1} = x_t
    # Тому x_{t+1} = M * x_t, де M = (I + gamma * A)^(-1)
    # Це матриця, яка діє на контур на кожному кроці.
    B = np.eye(N) + gamma * A
    M = inv(B)
    
    # --- 3. Обчислення власних чисел (Eigenvalues) ---
    # Власні числа показують, як множиться амплітуда кожної частоти за один крок.
    eigenvalues_M = eigvals(M)
    
    # Сортуємо їх (оскільки M симетрична/циклічна, числа будуть дійсними)
    # Але через чисельні похибки беремо модуль або реальну частину
    eigenvalues_M = np.sort(np.abs(eigenvalues_M))[::-1] # Від найбільшого до найменшого
    
    # --- 4. Візуалізація ---
    plt.figure(figsize=(12, 5))
    
    # Графік 1: Спектр власних чисел
    plt.subplot(1, 2, 1)
    plt.plot(eigenvalues_M, 'o-', markersize=3, label=f'alpha={alpha}, beta={beta}')
    plt.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Межа стабільності (1.0)')
    plt.title("Спектр власних чисел матриці еволюції M")
    plt.xlabel("Індекс частоти (k)")
    plt.ylabel("Власне число $\lambda_k$")
    plt.grid(True)
    plt.legend()
    
    # Графік 2: Швидкість загасання (Log scale)
    # Показує, як швидко зникає шум (високі частоти)
    plt.subplot(1, 2, 2)
    plt.semilogy(eigenvalues_M, 'o-', markersize=3, color='orange')
    plt.title("Логарифмічний масштаб (Log Scale)")
    plt.xlabel("Індекс частоти (k)")
    plt.ylabel("log($\lambda_k$)")
    plt.grid(True, which="both", ls="-")
    
    plt.tight_layout()
    plt.show()

    # --- 5. Текстовий висновок в консоль ---
    print(f"--- Аналіз для alpha={alpha}, beta={beta}, gamma={gamma} ---")
    print(f"Максимальне власне число: {eigenvalues_M[0]:.6f}")
    print(f"Мінімальне власне число:  {eigenvalues_M[-1]:.6f}")
    
    if np.all(eigenvalues_M <= 1.0000001):
        print("ВИСНОВОК: Система СТІЙКА (всі |lambda| <= 1).")
    else:
        print("ВИСНОВОК: Система НЕСТІЙКА (є |lambda| > 1).")
        
    print("Інтерпретація:")
    print("1. lambda ~ 1 відповідають низьким частотам (форма контуру зберігається).")
    print("2. lambda << 1 відповідають високим частотам (шум швидко гаситься).")

if __name__ == "__main__":
    # Експеримент 1: Стандартні параметри
    analyze_snake_operator(n_points=100, alpha=0.01, beta=0.1, gamma=1.0)
    
    # Експеримент 2: Більш жорстка змія (більше згладжування)
    # Розкоментуйте, щоб побачити різницю
    # analyze_snake_operator(n_points=100, alpha=0.1, beta=1.0, gamma=1.0)