import os

def rename_data(path):
    filenames = os.listdir(path)

    for i, filename in enumerate(filenames):
        old_file_path = os.path.join(path, filename)
        new_file_name = f"pick{i + 1}.jpg"
        new_file_path = os.path.join(path, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")

def main():
    dirs = ["data/healthy", "data/Rust"]
    for dir in dirs:
        rename_data(dir)

if __name__ == "__main__":
    main()
