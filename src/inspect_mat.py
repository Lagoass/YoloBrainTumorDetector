import os
import h5py

def inspect():
    mat_dir = "data/raw"
    files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat') and f != "cvind.mat"], key=lambda x: int(x.split('.')[0]))
    for f in files[:5]:
        path = os.path.join(mat_dir, f)
        try:
            with h5py.File(path, 'r') as f_in:
                # MATLAB saves struct as a group, typical format:
                # f_in['cjdata']
                cjdata = f_in['cjdata']
                label = int(cjdata['label'][0,0])
                pid = "".join([chr(c[0]) for c in cjdata['PID']]) if 'PID' in cjdata else str(cjdata['PID'][0,0])
                # h5py reads strings as arrays of ascii codes sometimes, let's see.
                image = cjdata['image'][:]
                tumorBorder = cjdata['tumorBorder'][:]
                print(f"File {f}: PID={pid}, Label={label}, ImageShape={image.shape}, BorderShape={tumorBorder.shape}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == '__main__':
    inspect()
