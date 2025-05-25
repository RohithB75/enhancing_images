import cv2
import os
import glob

def enhance_natural_colorful(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge and convert back to BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    final = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return final

def enhance_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

    for ext in image_extensions:
        for img_path in glob.glob(os.path.join(input_folder, ext)):
            try:
                enhanced = enhance_natural_colorful(img_path)
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, enhanced)
                print(f"Enhanced: {filename}")
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

# ---------- Run ----------
if __name__ == "__main__":
    input_folder = "input_images"      # Put your raw images here
    output_folder = "enhanced_images"  # Results will be saved here

    enhance_folder(input_folder, output_folder)
