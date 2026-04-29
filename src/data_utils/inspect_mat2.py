import os
import h5py
import numpy as np

def inspect():
    mat_dir = "data/raw"
    files = ["1.mat", "2.mat"]
    for f in files:
        path = os.path.join(mat_dir, f)
        with h5py.File(path, 'r') as f_in:
            cjdata = f_in['cjdata']
            image = cjdata['image'][:]
            tumorBorder = cjdata['tumorBorder'][:]
            print(f"File {f}: Image dtype={image.dtype}, min={image.min()}, max={image.max()}")
            print(f"tumorBorder shape: {tumorBorder.shape}")
            print(f"tumorBorder elements: {tumorBorder.flatten()[:10]}")

if __name__ == '__main__':
    inspect()
