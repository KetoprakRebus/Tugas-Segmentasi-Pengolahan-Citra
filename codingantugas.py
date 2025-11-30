import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==================== EDGE DETECTION METHODS ====================

def roberts_edge_detection(image):
    roberts_cross_x = np.array([[1, 0],
                                [0, -1]], dtype=np.float64)
    roberts_cross_y = np.array([[0, 1],
                                [-1, 0]], dtype=np.float64)
    
    # Konversi ke grayscale jika berwarna
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Konversi ke float untuk akurasi
    gray = gray.astype(np.float64)
    
    # Konvolusi dengan kernel Roberts
    grad_x = cv2.filter2D(gray, -1, roberts_cross_x)
    grad_y = cv2.filter2D(gray, -1, roberts_cross_y)
    
    # Hitung magnitude gradient
    gradient_magnitude = np.sqrt(grad_x*2 + grad_y*2)
    
    # Normalisasi ke range 0-255
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    return gradient_magnitude

def prewitt_edge_detection(image):
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float64)
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]], dtype=np.float64)
    
    # Konversi ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    # Konvolusi dengan kernel Prewitt
    grad_x = cv2.filter2D(gray, -1, prewitt_x)
    grad_y = cv2.filter2D(gray, -1, prewitt_y)
    
    # Hitung magnitude gradient
    gradient_magnitude = np.sqrt(grad_x*2 + grad_y*2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    return gradient_magnitude

def sobel_edge_detection(image):
    # Konversi ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    # Sobel gradient X dan Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Hitung magnitude gradient
    gradient_magnitude = np.sqrt(sobel_x*2 + sobel_y*2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    return gradient_magnitude


def frei_chen_edge_detection(image):
    sqrt2 = np.sqrt(2.0)
    
    frei_chen_x = np.array([[-1.0, 0.0, 1.0],
                            [-sqrt2, 0.0, sqrt2],
                            [-1.0, 0.0, 1.0]], dtype=np.float64)
    frei_chen_y = np.array([[-1.0, -sqrt2, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, sqrt2, 1.0]], dtype=np.float64)
    
    # Konversi ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    
    # Konvolusi dengan kernel Frei-Chen
    grad_x = cv2.filter2D(gray, -1, frei_chen_x)
    grad_y = cv2.filter2D(gray, -1, frei_chen_y)
    
    # Hitung magnitude gradient
    gradient_magnitude = np.sqrt(grad_x*2 + grad_y*2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    return gradient_magnitude

# ==================== NOISE FUNCTIONS ====================

def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    
    # Salt (putih)
    num_salt = int(np.ceil(amount * image.size * 0.5))
    coords = tuple([np.random.randint(0, i, num_salt) for i in image.shape])
    noisy[coords] = 255
    
    # Pepper (hitam)
    num_pepper = int(np.ceil(amount * image.size * 0.5))
    coords = tuple([np.random.randint(0, i, num_pepper) for i in image.shape])
    noisy[coords] = 0
    return noisy

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float64)
    noisy = image.astype(np.float64) + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# ==================== MAIN PROGRAM ====================
def main():
    print("="*80)
    print(" "*20 + "PROGRAM SEGMENTASI CITRA")
    print(" "*15 + "Metode: Roberts, Prewitt, Sobel, Frei-Chen")
    print("="*80)
    
    # ========== BAGIAN 1: BACA GAMBAR ==========
    image_filename = "hij.jpeg"  # <-- GANTI INI
    print(f"\nMembaca gambar: {image_filename}")
    img_original = cv2.imread(image_filename)
    
    if img_original is None:
        print(f"✗ Error: Gagal membaca '{image_filename}'")
        print(f"  Pastikan file ada di folder yang sama dengan script ini")
        return
    
    print(f"✓ Gambar berhasil dibaca")
    print(f"  Ukuran: {img_original.shape[1]}x{img_original.shape[0]} pixels")
    
    # ========== BAGIAN 2: SIAPKAN 4 CITRA ==========
    print("\nMempersiapkan 4 citra...")
    
    # 1. Citra Original (Color BGR)
    img_color_bgr = img_original.copy()
    
    # 2. Citra Grayscale
    img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    print("  ✓ Citra grayscale dibuat")
    
    # 3. Citra dengan derau Salt and Pepper
    img_salt_pepper = add_salt_pepper_noise(img_grayscale, amount=0.05)
    print("  ✓ Derau salt and pepper ditambahkan")
    
    # 4. Citra dengan derau Gaussian
    img_gaussian = add_gaussian_noise(img_grayscale, mean=0, sigma=25)
    print("  ✓ Derau Gaussian ditambahkan")
    
    # Dictionary untuk menyimpan 4 citra
    images = {
        'Original Color': img_color_bgr,
        'Grayscale': img_grayscale,
        'Salt & Pepper Noise': img_salt_pepper,
        'Gaussian Noise': img_gaussian
    }
    
    # ========== BAGIAN 3: PROSES SEGMENTASI ==========
    print("\nMemproses segmentasi dengan 4 metode...")
    print("-" * 80)
    
    # List metode edge detection
    methods = {
        'Roberts': roberts_edge_detection,
        'Prewitt': prewitt_edge_detection,
        'Sobel': sobel_edge_detection,
        'Frei-Chen': frei_chen_edge_detection
    }
    # Proses setiap citra dengan semua metode
    for img_name, img_data in images.items():
        print(f"\n→ Memproses: {img_name}"
              
        # Siapkan figure untuk 1 citra dengan 4 metode
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Segmentasi Citra: {img_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Subplot untuk gambar original
        ax = plt.subplot(2, 3, 1)
        if len(img_data.shape) == 3:
            # Konversi BGR ke RGB untuk display
            display_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            ax.imshow(display_img)
        else:
            ax.imshow(img_data, cmap='gray')
        ax.set_title('Original', fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')
        
        # Proses dengan 4 metode edge detection
        subplot_positions = [2, 3, 5, 6]  # Posisi untuk 4 hasil
        
        for idx, (method_name, method_func) in enumerate(methods.items()):
            print(f"  - {method_name}...", end=" ")
            
            # Terapkan metode edge detection
            result = method_func(img_data)
            
            # Tampilkan hasil
            ax = plt.subplot(2, 3, subplot_positions[idx])
            ax.imshow(result, cmap='gray')
            ax.set_title(method_name, fontsize=12, pad=10)
            ax.axis('off')
            
            print("✓")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Tampilkan window (non-blocking untuk citra berikutnya)
        plt.draw()
        plt.pause(0.1)
        
    print("\n" + "="*80)
    print("✓ PROSES SELESAI!")
    print("  Semua window telah ditampilkan.")
    print("  Tekan ENTER untuk menutup semua window...")
    print("="*80)
    
    # Tampilkan semua window dan tunggu input
    plt.show(block=False)
    input()  # Tunggu user menekan ENTER
    plt.close('all')
    
# ==================== EKSEKUSI PROGRAM ====================

if _name_ == "_main_":
    main()