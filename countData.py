import os

def count_files_in_directory(directory_path):
    try:
        items = os.listdir(directory_path)
        file_count = 0

        for item in items:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                file_count += 1

        return file_count

    except Exception as e:
        print(f"Error: {e}")
        return 0

directory_path_rust = 'data/Rust'
directory = ['data/Rust', 'data/Rust90',  'data/Rust180','data/Rust270', "data/healthy", "data/healthy90", "data/healthy180", "data/healthy270"]
total= 0
for path in directory:
    tmp = count_files_in_directory(path)
    print(f"Amount of files is: {tmp}")
    total += tmp

print(f"Total: {total}")