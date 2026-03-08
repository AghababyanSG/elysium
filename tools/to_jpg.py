from PIL import Image
import sys

def convert_to_jpg(input_path, output_path):
    img = Image.open(input_path)
    img = img.convert('RGB')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img.save(output_path, 'JPEG', quality=95)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python to_jpg.py <input_image> <output_image.jpg>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_to_jpg(input_path, output_path)
    print(f"Converted {input_path} -> {output_path}")

