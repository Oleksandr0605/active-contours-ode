import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

class Snake:
    def __init__(self, n_points=100, center=(0.5, 0.5), radius=0.3, alpha=0.01, beta=0.1, gamma=0.1):
        self.n_points = n_points
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        s = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = center[0] + radius * np.cos(s)
        y = center[1] + radius * np.sin(s)
        
        self.points = np.column_stack([x, y])
        
        self.A_inv = self._create_internal_matrix()

    def _create_internal_matrix(self):
        N = self.n_points
        alpha = self.alpha
        beta = self.beta
        
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
            
        B = np.eye(N) + self.gamma * A
        
        return inv(B)

    def add_noise(self, noise_level=0.01):
        noise = np.random.normal(0, noise_level, self.points.shape)
        self.points += noise

    def step(self):
        self.points = self.A_inv @ self.points

    def get_points(self):
        return np.vstack([self.points, self.points[0]])


if __name__ == "__main__":
    snake = Snake(n_points=100, alpha=0.05, beta=0.5, gamma=1.0)
        
    snake.add_noise(noise_level=0.05)
    
    plt.figure(figsize=(8, 8))
    plt.title("Еволюція активного контуру (тільки внутрішні сили)")
    
    pts = snake.get_points()
    plt.plot(pts[:, 0], pts[:, 1], 'r--', label='Початковий (шум)')
    
    for t in range(1, 51):
        snake.step()
        
        if t % 10 == 0:
            pts = snake.get_points()
            alpha_val = t / 50.0  
            plt.plot(pts[:, 0], pts[:, 1], 'b', alpha=alpha_val, label=f'Крок {t}')

    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()