import numpy as np

arr = np.array([2,7,11,15])
target = 9

for i in range(len(arr)):
    for j in range(i+1, len(arr)):
        if arr[i] + arr[j] == target:
            print("Indices:", i, j)



import numpy as np

arr = np.array([1,2,4,5])   
n = len(arr) + 1

expected_sum = n * (n+1) // 2
actual_sum = np.sum(arr)

missing = expected_sum - actual_sum
print("Missing Number:", missing)




import numpy as np

matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

rotated = np.rot90(matrix, k=-1)   
print(rotated)



import numpy as np

matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

main_sum = np.trace(matrix)                  # Main diagonal
sec_sum = np.trace(np.fliplr(matrix))        # Secondary diagonal
total = main_sum + sec_sum

print("Main:", main_sum)
print("Secondary:", sec_sum)
print("Total:", total)




# file: numpy_image_processing.py
import numpy as np
from PIL import Image
import os

def load_image_as_array(path):
    """Load image from path and return as uint8 numpy array (H, W, C)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr

def save_array_as_image(arr, path):
    """Save numpy array (H,W) or (H,W,C) as image."""
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)

def to_grayscale(arr):
    """
    Convert RGB array to grayscale using luminosity method.
    arr: (H, W, 3) uint8 -> returns (H, W) uint8
    """
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Input must be (H, W, 3) RGB array")
    # coefficients for R,G,B
    coeffs = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(arr[..., :3], coeffs)
    # clip and convert
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def change_brightness(arr, factor):
    """
    Change brightness by multiplying and clipping.
    arr can be grayscale (H,W) or RGB (H,W,3).
    factor >1 increases brightness, <1 decreases.
    """
    res = arr.astype(np.float32) * factor
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res

def process_image_example(image_path=None, out_prefix="out", make_gray=True, brightness_factor=1.2):
    """
    Example pipeline:
      - load image (or create sample)
      - convert to grayscale (optional)
      - change brightness
      - save outputs
    """
    # If no path provided, generate a sample gradient image
    if image_path is None or not os.path.exists(image_path):
        H, W = 256, 384
        # create RGB gradient with some pattern
        x = np.linspace(0, 255, W, dtype=np.uint8)
        y = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
        r = x[np.newaxis, :]
        g = y
        b = ((x + y) // 2).astype(np.uint8)
        sample = np.stack([r, g, b], axis=2)
        arr = sample
        print("No image path found — using generated sample image.")
    else:
        arr = load_image_as_array(image_path)
        print(f"Loaded image: {image_path} shape={arr.shape}")

    # Save original preview
    save_array_as_image(arr, f"{out_prefix}_original.png")
    print(f"Saved {out_prefix}_original.png")

    # Grayscale
    if make_gray:
        gray = to_grayscale(arr)
        save_array_as_image(gray, f"{out_prefix}_gray.png")
        print(f"Saved {out_prefix}_gray.png")
        bright_gray = change_brightness(gray, brightness_factor)
        save_array_as_image(bright_gray, f"{out_prefix}_gray_bright.png")
        print(f"Saved {out_prefix}_gray_bright.png")
    else:
        bright_rgb = change_brightness(arr, brightness_factor)
        save_array_as_image(bright_rgb, f"{out_prefix}_rgb_bright.png")
        print(f"Saved {out_prefix}_rgb_bright.png")

if __name__ == "__main__":
    # Example usage:
    # 1) with your image: python numpy_image_processing.py  (then edit process_image_example call)
    # 2) without image: it will create sample images in current folder
    process_image_example(image_path=None, out_prefix="sample", make_gray=True, brightness_factor=1.3)
