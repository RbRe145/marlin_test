import os
import numpy as np

def inspect_npy_directory(dir_path):
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir_path, filename)
            arr = np.load(file_path, allow_pickle=True)
            print(f"{filename}: dtype={arr.dtype}, shape={arr.shape}")

if __name__ == "__main__":
    inspect_npy_directory("/root/vllm/gen_test_data/moe_int4_awq_data_gpu")