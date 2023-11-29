import h5py

def print_structure(name, obj):
    """ HDF5ファイル内のオブジェクトの構造を出力する関数 """
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  Dataset {obj.shape} {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print("  Group")
    print()

def explore_hdf5(file_path):
    """ 指定されたHDF5ファイルの構造を出力する関数 """
    with h5py.File(file_path, "r") as file:
        print(f"Exploring {file_path}...\n")
        file.visititems(print_structure)

# ここにHDF5ファイルのパスを指定
file_path = 'datasets/crafter0.hdf5'
explore_hdf5(file_path)
