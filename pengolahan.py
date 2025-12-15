import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ===============================
# 1. KONFIGURASI PARAMETER
# ===============================

# Nama file gambar
IMAGE_PORTRAIT = "hij.jpeg"
IMAGE_LANDSCAPE = "jih.jpeg"

# Parameter noise
GAUSSIAN_VARIANCE_LOW = 0.005
GAUSSIAN_VARIANCE_HIGH = 0.05
SALT_PEPPER_DENSITY_LOW = 0.02
SALT_PEPPER_DENSITY_HIGH = 0.1

# Ukuran window filter (harus ganjil: 3, 5, 7, dst)
WINDOW_SIZE = 3

# ===============================
# 2. FUNGSI UTILITAS
# ===============================

def load_and_resize_image(image_path, width=400):
    img = Image.open(image_path)
    aspect_ratio = img.height / img.width
    new_height = int(width * aspect_ratio)
    img = img.resize((width, new_height), Image.LANCZOS)
    
    img_array = np.array(img)
    
    # Hapus channel alpha jika ada (RGBA -> RGB)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    return img_array

def rgb_to_grayscale(img):
    if len(img.shape) == 2:
        return img
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def list_image_files():
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    return [f for f in os.listdir('.') if f.lower().endswith(extensions)]

# ===============================
# 3. FUNGSI PENAMBAHAN NOISE
# ===============================

def add_gaussian_noise(img, variance):
    # Normalisasi ke [0, 1]
    img_float = img.astype(np.float64) / 255.0
    std_dev = np.sqrt(variance)
    
    # Generate noise dari distribusi normal
    noise = np.random.normal(0, std_dev, img_float.shape)
    
    # Tambahkan noise
    noisy_img = img_float + noise
    
    # Clip ke [0, 1] dan konversi ke uint8
    noisy_img = np.clip(noisy_img, 0, 1)
    return (noisy_img * 255).astype(np.uint8)

def add_salt_pepper_noise(img, density):
    noisy_img = img.copy()
    random_matrix = np.random.random(img.shape[:2])
    
    # Salt: density/2 piksel putih
    salt_mask = random_matrix < (density / 2)
    # Pepper: density/2 piksel hitam
    pepper_mask = random_matrix > (1 - density / 2)
    
    if len(img.shape) == 3:  # RGB
        for i in range(img.shape[2]):
            noisy_img[:, :, i][salt_mask] = 255
            noisy_img[:, :, i][pepper_mask] = 0
    else:  # Grayscale
        noisy_img[salt_mask] = 255
        noisy_img[pepper_mask] = 0
    
    return noisy_img

# ===============================
# 4. FUNGSI FILTER
# ===============================

def add_padding(img, pad_size):
    if len(img.shape) == 3:  # RGB
        padded = np.pad(img, 
                       ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
                       mode='edge')
    else:  # Grayscale
        padded = np.pad(img, 
                       ((pad_size, pad_size), (pad_size, pad_size)), 
                       mode='edge')
    return padded

def mean_filter(img, window_size=3):
    pad_size = window_size // 2
    padded_img = add_padding(img, pad_size)
    
    if len(img.shape) == 3:  # RGB
        filtered = np.zeros_like(img, dtype=np.float64)
        height, width, channels = img.shape
        
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    window = padded_img[i:i+window_size, j:j+window_size, c]
                    filtered[i, j, c] = np.mean(window)
    else:  # Grayscale
        filtered = np.zeros_like(img, dtype=np.float64)
        height, width = img.shape
        
        for i in range(height):
            for j in range(width):
                window = padded_img[i:i+window_size, j:j+window_size]
                filtered[i, j] = np.mean(window)
    
    return filtered.astype(np.uint8)

def median_filter(img, window_size=3):
    pad_size = window_size // 2
    padded_img = add_padding(img, pad_size)
    
    if len(img.shape) == 3:  # RGB
        filtered = np.zeros_like(img, dtype=np.uint8)
        height, width, channels = img.shape
        
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    window = padded_img[i:i+window_size, j:j+window_size, c]
                    filtered[i, j, c] = np.median(window)
    else:  # Grayscale
        filtered = np.zeros_like(img, dtype=np.uint8)
        height, width = img.shape
        
        for i in range(height):
            for j in range(width):
                window = padded_img[i:i+window_size, j:j+window_size]
                filtered[i, j] = np.median(window)
    
    return filtered

def min_filter(img, window_size=3):
    pad_size = window_size // 2
    padded_img = add_padding(img, pad_size)
    
    if len(img.shape) == 3:  # RGB
        filtered = np.zeros_like(img, dtype=np.uint8)
        height, width, channels = img.shape
        
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    window = padded_img[i:i+window_size, j:j+window_size, c]
                    filtered[i, j, c] = np.min(window)
    else:  # Grayscale
        filtered = np.zeros_like(img, dtype=np.uint8)
        height, width = img.shape
        
        for i in range(height):
            for j in range(width):
                window = padded_img[i:i+window_size, j:j+window_size]
                filtered[i, j] = np.min(window)
    
    return filtered

def max_filter(img, window_size=3):
    pad_size = window_size // 2
    padded_img = add_padding(img, pad_size)
    
    if len(img.shape) == 3:  # RGB
        filtered = np.zeros_like(img, dtype=np.uint8)
        height, width, channels = img.shape
        
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    window = padded_img[i:i+window_size, j:j+window_size, c]
                    filtered[i, j, c] = np.max(window)
    else:  # Grayscale
        filtered = np.zeros_like(img, dtype=np.uint8)
        height, width = img.shape
        
        for i in range(height):
            for j in range(width):
                window = padded_img[i:i+window_size, j:j+window_size]
                filtered[i, j] = np.max(window)
    
    return filtered

# ===============================
# 5. FUNGSI EVALUASI
# ===============================

def calculate_mse(original, processed):
    original_float = original.astype(np.float64)
    processed_float = processed.astype(np.float64)
    
    squared_diff = (original_float - processed_float) ** 2
    mse = np.mean(squared_diff)
    
    return mse

# ===============================
# 6. FUNGSI PROSES & VISUALISASI
# ===============================

def process_image(img, noise_type, noise_param, window_size=3):
    # Tambahkan noise
    if noise_type == "Gaussian":
        noisy_img = add_gaussian_noise(img, noise_param)
    else:  # Salt & Pepper
        noisy_img = add_salt_pepper_noise(img, noise_param)
    
    # Aplikasikan semua filter
    filtered_mean = mean_filter(noisy_img, window_size)
    filtered_median = median_filter(noisy_img, window_size)
    filtered_min = min_filter(noisy_img, window_size)
    filtered_max = max_filter(noisy_img, window_size)
    
    # Hitung MSE untuk semua hasil
    results = {
        'noisy': noisy_img,
        'mean': filtered_mean,
        'median': filtered_median,
        'min': filtered_min,
        'max': filtered_max,
        'mse': {
            'noise': calculate_mse(img, noisy_img),
            'mean': calculate_mse(img, filtered_mean),
            'median': calculate_mse(img, filtered_median),
            'min': calculate_mse(img, filtered_min),
            'max': calculate_mse(img, filtered_max)
        }
    }
    
    return results

def visualize_results(original, results, noise_type, noise_param, img_type):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{img_type} - {noise_type} Noise ({noise_param})', 
                 fontsize=16, fontweight='bold')
    
    cmap = 'gray' if len(original.shape) == 2 else None
    
    # Plot semua gambar
    images = [
        (original, 'Original', (0, 0)),
        (results['noisy'], f"Noisy\nMSE: {results['mse']['noise']:.2f}", (0, 1)),
        (results['mean'], f"Mean Filter\nMSE: {results['mse']['mean']:.2f}", (0, 2)),
        (results['median'], f"Median Filter\nMSE: {results['mse']['median']:.2f}", (1, 0)),
        (results['min'], f"Min Filter\nMSE: {results['mse']['min']:.2f}", (1, 1)),
        (results['max'], f"Max Filter\nMSE: {results['mse']['max']:.2f}", (1, 2))
    ]
    
    for img_data, title, (row, col) in images:
        axes[row, col].imshow(img_data, cmap=cmap)
        axes[row, col].set_title(title, fontsize=11, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

# ===============================
# 7. MAIN PROGRAM
# ===============================

def main():
    print("=" * 80)
    print(" " * 25 + "PROGRAM RESTORASI CITRA DIGITAL")
    print("=" * 80)
    
    # Tampilkan file yang tersedia
    available_images = list_image_files()
    if available_images:
        print(f"\n📁 File gambar tersedia: {', '.join(available_images[:5])}")
        if len(available_images) > 5:
            print(f"    ... dan {len(available_images) - 5} file lainnya")
    
    # Tampilkan konfigurasi
    print(f"\n⚙ Konfigurasi:")
    print(f"   Portrait     : {IMAGE_PORTRAIT}")
    print(f"   Landscape    : {IMAGE_LANDSCAPE}")
    print(f"   Window Size  : {WINDOW_SIZE}x{WINDOW_SIZE}")
    print(f"   Gaussian     : {GAUSSIAN_VARIANCE_LOW}, {GAUSSIAN_VARIANCE_HIGH}")
    print(f"   Salt&Pepper  : {SALT_PEPPER_DENSITY_LOW}, {SALT_PEPPER_DENSITY_HIGH}")
    
    # Load gambar
    images_to_process = []
    
    if os.path.exists(IMAGE_PORTRAIT):
        print(f"\n✓ Loading {IMAGE_PORTRAIT}...")
        p_rgb = load_and_resize_image(IMAGE_PORTRAIT)
        p_gray = rgb_to_grayscale(p_rgb)
        images_to_process.extend([
            (p_rgb, "Portrait RGB"),
            (p_gray, "Portrait Grayscale")
        ])
    else:
        print(f"\n✗ File {IMAGE_PORTRAIT} tidak ditemukan!")
    
    if os.path.exists(IMAGE_LANDSCAPE):
        print(f"✓ Loading {IMAGE_LANDSCAPE}...")
        l_rgb = load_and_resize_image(IMAGE_LANDSCAPE)
        l_gray = rgb_to_grayscale(l_rgb)
        images_to_process.extend([
            (l_rgb, "Landscape RGB"),
            (l_gray, "Landscape Grayscale")
        ])
    else:
        print(f"✗ File {IMAGE_LANDSCAPE} tidak ditemukan!")
    
    if not images_to_process:
        print("\n⚠ Tidak ada gambar yang dapat diproses!")
        return
    
    print("\n" + "=" * 80)
    print(" " * 28 + "MEMPROSES GAMBAR...")
    print("=" * 80)
    
    # Simpan semua hasil
    all_results = {}
    
    # Proses setiap gambar
    for img, img_type in images_to_process:
        print(f"\n→ Memproses: {img_type}...", end=" ", flush=True)
        
        all_results[img_type] = {}
        
        # Proses 4 kondisi noise
        conditions = [
            ("Gaussian", GAUSSIAN_VARIANCE_LOW, "gauss_low"),
            ("Gaussian", GAUSSIAN_VARIANCE_HIGH, "gauss_high"),
            ("Salt & Pepper", SALT_PEPPER_DENSITY_LOW, "sp_low"),
            ("Salt & Pepper", SALT_PEPPER_DENSITY_HIGH, "sp_high")
        ]
        
        for noise_type, param, key in conditions:
            results = process_image(img, noise_type, param, WINDOW_SIZE)
            all_results[img_type][key] = results
            visualize_results(img, results, noise_type, param, img_type)
        
        print("✓")
    
    # ===============================
    # TAMPILKAN RINGKASAN LENGKAP
    # ===============================
    
    print("\n" + "=" * 80)
    print(" " * 25 + "RINGKASAN HASIL MSE")
    print("=" * 80)
    
    for img_type, results_dict in all_results.items():
        print(f"\n{'=' * 80}")
        print(f"  {img_type.upper()}")
        print(f"{'=' * 80}")
        
        condition_names = {
            'gauss_low': f'Gaussian Noise (variance {GAUSSIAN_VARIANCE_LOW})',
            'gauss_high': f'Gaussian Noise (variance {GAUSSIAN_VARIANCE_HIGH})',
            'sp_low': f'Salt & Pepper Noise (density {SALT_PEPPER_DENSITY_LOW})',
            'sp_high': f'Salt & Pepper Noise (density {SALT_PEPPER_DENSITY_HIGH})'
        }
        
        for key, results in results_dict.items():
            print(f"\n  {condition_names[key]}:")
            mse = results['mse']
            
            # Tampilkan MSE dalam format tabel
            print(f"    {'Condition':<15} {'MSE Value':>12}")
            print(f"    {'-' * 29}")
            print(f"    {'Noise':<15} {mse['noise']:>12.4f}")
            print(f"    {'Mean Filter':<15} {mse['mean']:>12.4f}")
            print(f"    {'Median Filter':<15} {mse['median']:>12.4f}")
            print(f"    {'Min Filter':<15} {mse['min']:>12.4f}")
            print(f"    {'Max Filter':<15} {mse['max']:>12.4f}")
            
            # Tentukan filter terbaik
            filters = {
                'Mean': mse['mean'],
                'Median': mse['median'],
                'Min': mse['min'],
                'Max': mse['max']
            }
            best_filter = min(filters, key=filters.get)
            best_mse = filters[best_filter]
            
            print(f"\n    ⭐ Filter Terbaik: {best_filter} Filter (MSE: {best_mse:.4f})")   
    # Tunggu input user
    plt.show(block=False)
    input()
    plt.close('all')
    
    print("\n✓ Program selesai. Terima kasih!")

if __name__ == "__main__":
    main()