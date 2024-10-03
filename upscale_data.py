import os
import torch
import torchvision.io as io
from einops import rearrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def resize_scale(clip, target_size, interpolation_mode="bilinear"):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    H, W = clip.size(-2), clip.size(-1)
    scale_ = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(clip, scale_factor=scale_, mode=interpolation_mode, align_corners=False)


def resize_video(mp4_file, target_size=(128, 128), interpolation_mode="bilinear"):
    # 動画を読み込み
    video_tensor, _, info = io.read_video(mp4_file, pts_unit="sec")

    # 動画をリサイズ
    video_tensor = rearrange(video_tensor, "T H W C -> T C H W")
    resized_video = resize_scale(video_tensor, target_size, interpolation_mode)
    resized_video = rearrange(resized_video, "T C H W -> T H W C")

    # リサイズされた動画を同じファイル名で上書き保存
    io.write_video(mp4_file, resized_video, info["video_fps"])

    return mp4_file


def resize_videos_in_directory(file_path, target_size=(128, 128)):
    mp4_files = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))

    print(f"Found {len(mp4_files)} MP4 files")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(resize_video, mp4_file, target_size) for mp4_file in mp4_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing videos"):
            resized_file = future.result()
            print(f"Resized and overwritten video: {resized_file}")


if __name__ == "__main__":
    # Configurations
    file_path = "/data/expertdata/train"

    # Resize videos to 128x128
    resize_videos_in_directory(file_path)
