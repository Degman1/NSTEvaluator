import os
import sys
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def resize_and_center_crop_images(input_dir, output_dir, crop_size=512):
    """
    Resize and center crop all images in the input directory and save them to the output directory.
    
    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save cropped images.
        crop_size (int): Desired crop size (square, width=height).
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transforms: Resize to crop size first, then center crop
    transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop((crop_size, crop_size))
    ])

    for file_name in tqdm(os.listdir(input_dir), desc="Processing images"):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            print(f"Skipping non-image file: {file_name}")
            continue

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            with Image.open(input_path) as img:
                # Apply resize and center crop
                processed_img = transform(img)
                processed_img.save(output_path)
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to process image: {file_name}, Error: {e}")

if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python resize_and_center_crop_images.py <input_directory> <output_directory> [crop_size]")
        sys.exit(1)

    # Parse command-line arguments
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    crop_size = int(sys.argv[3]) if len(sys.argv) == 4 else 512

    # Run the resize and center crop function
    resize_and_center_crop_images(input_directory, output_directory, crop_size)
