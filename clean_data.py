import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def clean_files(file_path):
    for root, _, files in os.walk(file_path):
        for file in files:
            full_path = os.path.join(root, file)
            dir_name = os.path.basename(root)
            if file.endswith(".mp4"):
                json_file = full_path.replace(".mp4", ".json")
                if not os.path.exists(json_file):
                    print(f"Missing: {full_path}, {json_file}")
                    os.remove(full_path)

            elif file.endswith(".json"):
                mp4_file = full_path.replace(".json", ".mp4")
                if not os.path.exists(mp4_file):
                    print(f"Missing: {mp4_file}, {full_path}")
                    os.remove(full_path)


if __name__ == "__main__":
    # Configurations
    file_path = "/data/expertdata-long/train"

    # Find and delete unused files based on video count
    clean_files(file_path)
