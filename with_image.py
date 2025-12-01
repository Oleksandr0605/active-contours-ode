import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import inv
from scipy.interpolate import RectBivariateSpline
import os

class ExternalForce:
    def __init__(self, image, sigma=5.0):
        """
        Calculates the force field from the image.
        sigma: blur strength.
        """
        self.H, self.W = image.shape
        
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        self.fx = cv2.Sobel(magnitude, cv2.CV_64F, 1, 0, ksize=3)
        self.fy = cv2.Sobel(magnitude, cv2.CV_64F, 0, 1, ksize=3)
        
        force_mag = np.sqrt(self.fx**2 + self.fy**2)
        self.fx /= (force_mag + 1e-5)
        self.fy /= (force_mag + 1e-5)
        
        x_range = np.arange(self.W)
        y_range = np.arange(self.H)
        self.interp_fx = RectBivariateSpline(y_range, x_range, self.fx)
        self.interp_fy = RectBivariateSpline(y_range, x_range, self.fy)

    def get_force(self, points):
        """Returns force vectors (Fx, Fy) for given point coordinates"""
        fx_vals = self.interp_fx(points[:, 1], points[:, 0], grid=False)
        fy_vals = self.interp_fy(points[:, 1], points[:, 0], grid=False)
        return np.column_stack([fx_vals, fy_vals])

class Snake:
    def __init__(self, start_points, alpha=0.01, beta=0.1, gamma=1.0, w_line=10.0):
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
            
        return inv(np.eye(N) + self.gamma * A)

    def step(self, external_force_field):
        f_ext = external_force_field.get_force(self.points)
        
        total_force = self.gamma * self.w_line * f_ext
        self.points = self.A_inv @ (self.points + total_force)
        
        H, W = external_force_field.H, external_force_field.W
        self.points[:, 0] = np.clip(self.points[:, 0], 0, W-1)
        self.points[:, 1] = np.clip(self.points[:, 1], 0, H-1)

    def get_points(self):
        return np.vstack([self.points, self.points[0]])

def create_synthetic_image():
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 60, 255, -1)
    cv2.rectangle(img, (40, 40), (100, 100), 255, -1)
    return img

if __name__ == "__main__":
    filename = 'image2.png'
    if os.path.exists(filename):
        print(f"Loading {filename}...")
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error reading file. Using synthetic image.")
            img = create_synthetic_image()
    else:
        print(f"File {filename} not found. Using synthetic image.")
        img = create_synthetic_image()

    H, W = img.shape
    print(f"Image Dimensions: W={W}, H={H}")
    
    ext_force = ExternalForce(img, sigma=3.0)
    
    n_points = 150
    s = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    center_x, center_y = W / 2, H / 2 - 100
    radius = min(W, H) * 0.40 
    
    x = center_x + radius * np.cos(s)
    y = center_y + radius * np.sin(s)
    init_points = np.column_stack([x, y])
    
    snake = Snake(init_points, alpha=0.5, beta=0.5, gamma=1.0, w_line=8.0)    
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Force Field ({W}x{H})")
    plt.imshow(img, cmap='gray', alpha=0.5)
    
    step_x = max(1, W // 25) 
    step_y = max(1, H // 25)
    
    grid_x = np.arange(0, W, step_x)
    grid_y = np.arange(0, H, step_y)
    X, Y = np.meshgrid(grid_x, grid_y)
    
    FX = ext_force.interp_fx(Y, X, grid=False) 
    FY = ext_force.interp_fy(Y, X, grid=False)
    
    plt.quiver(X, Y, FX, FY, color='red', scale=30, alpha=0.6)
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.title("Active Contour Evolution")
    plt.imshow(img, cmap='gray')
    
    snake_line, = plt.plot([], [], 'g-', lw=2, label='Snake')
    plt.legend()
    
    steps = 1000
    for t in range(steps):
        snake.step(ext_force)
        
        if t % 50 == 0: 
            pts = snake.get_points()
            snake_line.set_data(pts[:, 0], pts[:, 1])
            plt.title(f"Step: {t}/{steps}")
            plt.pause(0.0001)
            
    plt.show()