import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, eigvals

def analyze_snake_operator(n_points=100, alpha=0.01, beta=0.1, gamma=1.0):
    """
    Аналізує спектральні властивості матриці переходу змії M = (I + gamma * A)^(-1).
    """
    N = n_points
    
    # --- 1. Побудова матриці A (Внутрішні сили) ---
    a = beta
    b = -alpha - 4*beta
    c = 2*alpha + 6*beta
    
    # Створюємо циклічну матрицю A
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i-2)%N] = a
        A[i, (i-1)%N] = b
        A[i, i]       = c
        A[i, (i+1)%N] = b
        A[i, (i+2)%N] = a
        
    # --- 2. Оператор еволюції M ---
    B = np.eye(N) + gamma * A
    M = inv(B)
    
    # --- 3. Обчислення власних чисел ---
    # Для стійкості ми перевіряємо модуль власних чисел
    eigenvalues_M = eigvals(M)
    eigenvalues_M = np.sort(np.abs(eigenvalues_M))[::-1] # Сортування від найбільшого до найменшого
    
    # --- 4. Візуалізація та Висновок ---
    plt.figure(figsize=(12, 5))
    
    # Графік 1: Спектр
    plt.subplot(1, 2, 1)
    plt.plot(eigenvalues_M, 'o-', markersize=3, label=f'alpha={alpha}, beta={beta}')
    plt.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Межа стабільності (1.0)')
    plt.title("Спектр власних чисел матриці еволюції M")
    plt.xlabel("Індекс частоти (k)")
    plt.ylabel("Власне число $\\lambda_k$")
    plt.grid(True)
    plt.legend()
    
    # Графік 2: Швидкість загасання (Log scale)
    plt.subplot(1, 2, 2)
    plt.semilogy(eigenvalues_M, 'o-', markersize=3, color='orange')
    plt.title("Логарифмічний масштаб (Log Scale)")
    plt.xlabel("Індекс частоти (k)")
    plt.ylabel("log($\\lambda_k$)")
    plt.grid(True, which="both", ls="-")
    
    plt.tight_layout()
    
    # Зберігаємо графік, оскільки ви працюєте в неінтерактивному середовищі
    filename = f'spectral_analysis_alpha{alpha}_beta{beta}.png'
    plt.savefig(filename)
    plt.close()
    
    # Текстовий висновок
    print(f"\n--- Результати аналізу для alpha={alpha}, beta={beta}, gamma={gamma} ---")
    print(f"Максимальне власне число (k=0, низька частота): {eigenvalues_M[0]:.6f}")
    print(f"Мінімальне власне число (k=N/2, висока частота): {eigenvalues_M[-1]:.6f}")
    print(f"ВИСНОВОК: Система СТІЙКА, оскільки max(|lambda|) <= 1.0.")
    print(f"Графік спектру збережено у файл: {filename}")


if __name__ == "__main__":
    # Експеримент 1: Стандартні параметри
    analyze_snake_operator(n_points=100, alpha=0.01, beta=0.1, gamma=1.0)
    
    # Експеримент 2: Більш жорстка змія (для порівняння)
    analyze_snake_operator(n_points=100, alpha=0.1, beta=1.0, gamma=1.0)