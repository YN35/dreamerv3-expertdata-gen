from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    folder_path="/data/mimic_arena_gz",
    # path_in_repo="",
    repo_id="ramu0e/mimic-arena",
    repo_type="dataset",
)
