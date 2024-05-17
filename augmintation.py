import os
from PIL import Image

def augmintate_data(directory_path: str, dest_dir: str, rotate_degree: int) -> None:
    try:
        os.makedirs(dest_dir, exist_ok=True)

        items = os.listdir(directory_path)
        for filename in items:
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path) and filename.endswith(('.jpg', '.jpeg', '.png')):

                image = Image.open(file_path)

                rotated_image = image.rotate(rotate_degree, expand=True)
                rotated_file_path = os.path.join(dest_dir, filename)

                rotated_image.save(rotated_file_path)
                print(f"Image is created: {rotated_file_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    plant_type = ["data/healthy",  "data/Rust"]
    augmentation_degree =  [90, 180, 270]

    for type in plant_type:
        for rotate in augmentation_degree:
            dest_dir = f"{type}{rotate}"
            print("Target dir:", dest_dir)
            augmintate_data(type, dest_dir, rotate)

if __name__ == "__main__":
    main() 
