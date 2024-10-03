import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def count_video_files_in_directory(file_path, directory_epsd_len):
    directory_video_count = {dir_path: 0 for dir_path in directory_epsd_len}

    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".mp4"):
                dir_name = os.path.basename(root)
                if dir_name in directory_video_count:
                    directory_video_count[dir_name] += 1

    return directory_video_count


def find_unused_files(file_path, directory_epsd_len, directory_video_count):
    used_files = set()

    # Determine how many video files should be kept based on the `directory_epsd_len`
    directory_keep_count = {dir_name: min(count, directory_epsd_len[dir_name]) for dir_name, count in directory_video_count.items()}

    for root, _, files in os.walk(file_path):
        for file in files:
            full_path = os.path.join(root, file)
            dir_name = os.path.basename(root)
            if file.endswith(".mp4") and dir_name in directory_keep_count:
                if directory_keep_count[dir_name] > 0:
                    directory_keep_count[dir_name] -= 1
                    used_files.add(full_path)

    unused_files = []
    for root, _, files in os.walk(file_path):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith(".mp4") and full_path not in used_files:
                json_file = full_path.replace(".mp4", ".json")
                print(f"Unused: {full_path}, {json_file}")
                os.remove(full_path)
                os.remove(json_file)
                print(f"Deleted: {full_path}, {json_file}")

    return unused_files


if __name__ == "__main__":
    # Configurations
    file_path = "/data/expertdata-long/val"
    # directory_epsd_len = {
    #     "atari_ms_pacman": 30000,  # Expected number of videos
    #     "atari_pong": 20000,
    #     "crafter": 100000,
    #     "dmc_cartpole_balance": 15000,
    #     "dmc_walker_walk": 30000,
    # }
    directory_epsd_len = {
        "atari_ms_pacman": 300,  # Expected number of videos
        "atari_pong": 200,
        "crafter": 1000,
        "dmc_cartpole_balance": 150,
        "dmc_walker_walk": 300,
    }

    # Count video files in each directory
    directory_video_count = count_video_files_in_directory(file_path, directory_epsd_len)

    # Display the number of videos in each directory
    for dir_name, count in directory_video_count.items():
        print(f"Directory: {dir_name}, Video files: {count}")

    # Find and delete unused files based on video count
    find_unused_files(file_path, directory_epsd_len, directory_video_count)
