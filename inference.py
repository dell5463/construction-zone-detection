import os
from ultralytics import YOLO
import cv2

# --------------------------
# CONFIGURATION
# --------------------------
MODEL_PATH = "C:/Users/chakk/Desktop/CSODYOLO/runs/detect/train3/weights/best.pt"
IMAGE_DIR = "randimg/"            # folder containing test images
OUTPUT_DIR = "inference_output/"  # where predictions will be saved
# --------------------------

def main():
    # Load trained model
    model = YOLO(MODEL_PATH)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all image files in the folder
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(exts)]

    if not image_files:
        print("No images found in:", IMAGE_DIR)
        return

    print(f"Found {len(image_files)} images. Running inference...")

    # Iterate over every image
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        print("Processing:", img_path)

        # Run inference
        results = model(img_path)

        # Save output
        output_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
        results[0].save(filename=output_path)

        print("Saved:", output_path)

    print("Batch inference complete.")


if __name__ == "__main__":
    main()
