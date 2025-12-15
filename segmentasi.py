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
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
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
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
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
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
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
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
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

# ==================== MSE CALCULATION ====================

def calculate_mse(original, processed):

    if original.shape != processed.shape:
        # Resize processed image jika diperlukan
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Konversi ke float untuk perhitungan MSE
    original_float = original.astype(np.float64)
    processed_float = processed.astype(np.float64)
    
    # Hitung MSE
    mse_value = np.mean((original_float - processed_float) ** 2)
    
    return mse_value

def calculate_mse_between_methods(method_results):
    methods = list(method_results.keys())
    mse_matrix = {}
    
    for i, method1 in enumerate(methods):
        mse_matrix[method1] = {}
        for j, method2 in enumerate(methods):
            if i != j:
                mse_value = calculate_mse(method_results[method1], method_results[method2])
                mse_matrix[method1][method2] = mse_value
            else:
                mse_matrix[method1][method2] = 0.0
    
    return mse_matrix

# ==================== MAIN PROGRAM ====================
def main():
    print("="*80)
    print(" "*20 + "PROGRAM SEGMENTASI CITRA")
    print(" "*15 + "Metode: Roberts, Prewitt, Sobel, Frei-Chen")
    print(" "*10 + "Dengan Perhitungan MSE (Mean Squared Error)")
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
    
    # 2. Citra Grayscale (akan digunakan sebagai referensi untuk MSE)
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
    
    # Dictionary untuk menyimpan semua hasil
    all_results = {}
    all_mse_values = {}
    
    # Proses setiap citra dengan semua metode
    for img_name, img_data in images.items():
        print(f"\n→ Memproses: {img_name}")
        
        # Dictionary untuk menyimpan hasil metode pada citra ini
        method_results = {}
        mse_values = {}
        
        # Skip MSE untuk citra berwarna karena perbandingannya akan berbeda
        if img_name == 'Original Color':
            # Untuk citra berwarna, gunakan grayscale sebagai referensi
            reference_img = img_grayscale
        else:
            reference_img = img_data if len(img_data.shape) == 2 else cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        
        # Siapkan figure untuk 1 citra dengan 4 metode
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Segmentasi Citra: {img_name}\nDengan Nilai MSE', 
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
            method_results[method_name] = result
            
            # Hitung MSE terhadap citra referensi (grayscale original)
            mse_value = calculate_mse(reference_img, result)
            mse_values[method_name] = mse_value
            
            # Tampilkan hasil dengan MSE di judul
            ax = plt.subplot(2, 3, subplot_positions[idx])
            ax.imshow(result, cmap='gray')
            
            # Format nilai MSE (2 desimal)
            mse_formatted = f"{mse_value:.2f}"
            ax.set_title(f'{method_name}\nMSE: {mse_formatted}', 
                         fontsize=12, pad=10)
            ax.axis('off')
            
            print(f"✓ (MSE: {mse_formatted})")
        
        # Simpan hasil
        all_results[img_name] = method_results
        all_mse_values[img_name] = mse_values
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Tampilkan window (non-blocking untuk citra berikutnya)
        plt.draw()
        plt.pause(0.1)
    
    # ========== BAGIAN 4: TAMPILKAN TABEL PERBANDINGAN MSE ==========
    print("\n" + "="*80)
    print("PERBANDINGAN NILAI MSE ANTARA METODE")
    print("="*80)
    
    # Tampilkan MSE untuk setiap jenis citra
    for img_name, mse_dict in all_mse_values.items():
        print(f"\nCitra: {img_name}")
        print("-" * 40)
        for method_name, mse_value in mse_dict.items():
            print(f"  {method_name:10} : {mse_value:8.2f}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Hitung rata-rata MSE untuk setiap metode
    print("\nRata-rata MSE untuk setiap metode:")
    print("-" * 40)
    
    # Kumpulkan semua nilai MSE per metode
    method_mse_totals = {}
    method_mse_counts = {}
    
    for img_name, mse_dict in all_mse_values.items():
        for method_name, mse_value in mse_dict.items():
            if method_name not in method_mse_totals:
                method_mse_totals[method_name] = 0
                method_mse_counts[method_name] = 0
            method_mse_totals[method_name] += mse_value
            method_mse_counts[method_name] += 1
    
    # Hitung dan tampilkan rata-rata
    for method_name in method_mse_totals.keys():
        avg_mse = method_mse_totals[method_name] / method_mse_counts[method_name]
        print(f"  {method_name:10} : {avg_mse:8.2f} (rata-rata)")
    
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

if __name__ == "__main__":
    main()