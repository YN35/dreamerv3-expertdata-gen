import os
import uuid
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def rename_file_pair(json_path, mp4_path):
    # Generate a new UID for the file names
    new_uid = str(uuid.uuid4())

    # Construct new file paths
    new_json_path = os.path.join(os.path.dirname(json_path), f"{new_uid}.json")
    new_mp4_path = os.path.join(os.path.dirname(mp4_path), f"{new_uid}.mp4")

    # Rename the JSON and MP4 files
    shutil.move(json_path, new_json_path)
    shutil.move(mp4_path, new_mp4_path)

    return json_path, new_json_path, mp4_path, new_mp4_path


def rename_files_with_uid_parallel(file_path):
    # Find all JSON and MP4 files
    files_to_rename = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                mp4_path = json_path.replace(".json", ".mp4")
                if os.path.exists(mp4_path):
                    files_to_rename.append((json_path, mp4_path))

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(rename_file_pair, json_path, mp4_path) for json_path, mp4_path in files_to_rename]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Renaming files"):
            json_path, new_json_path, mp4_path, new_mp4_path = future.result()
            print(f"Renamed:\n{json_path} -> {new_json_path}\n{mp4_path} -> {new_mp4_path}")


if __name__ == "__main__":
    # Configuration
    file_path = "/data/expertdata-long/val"

    # Rename files in parallel
    rename_files_with_uid_parallel(file_path)
