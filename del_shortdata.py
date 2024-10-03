import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def check_and_delete_video_with_json(mp4_file, min_frames):
    # MP4ファイルのフレーム数を取得
    cap = cv2.VideoCapture(mp4_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # フレーム数が100未満の場合、MP4ファイルと対応するJSONファイルを削除
    if frame_count < min_frames:
        json_file = os.path.splitext(mp4_file)[0] + ".json"
        os.remove(mp4_file)
        os.remove(json_file)
        return mp4_file, json_file, frame_count
    return None


def delete_short_videos(file_path, min_frames=64):
    mp4_files = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))

    print(f"Found {len(mp4_files)} MP4 files")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(check_and_delete_video_with_json, mp4_file, min_frames) for mp4_file in mp4_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Deleting short videos and related JSON files"):
            result = future.result()
            if result is not None:
                mp4_file, json_file, frame_count = result
                print(f"Deleted: {mp4_file} ({frame_count} frames), {json_file}")


if __name__ == "__main__":
    # Configurations
    file_path = "/data/expertdata-long/val"

    # Delete videos with less than 100 frames
    delete_short_videos(file_path)
