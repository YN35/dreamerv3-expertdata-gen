from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    folder_path="models",
    # path_in_repo="",
    repo_id="ramu0e/dreamerv3-models",
    repo_type="model",
)
