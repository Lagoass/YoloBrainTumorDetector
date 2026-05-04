import os
import glob
import h5py
import numpy as np
import cv2
from tqdm import tqdm

def process_mat_file(path):
    """
    Reads a .mat file and extracts:
    - PID (string)
    - label (int): mapped to 0, 1, 2
    - image (numpy array HxW)
    - tumorBorder (numpy array)
    """
    with h5py.File(path, 'r') as f_in:
        cjdata = f_in['cjdata']
        
        # Parse label (1=meningioma, 2=glioma, 3=pituitary tumor) -> (0, 1, 2)
        label = int(cjdata['label'][0, 0]) - 1
        
        # Parse PID
        pid = cjdata['PID']
        # PID can be stored as an array of characters (ascii codes) or a single number string
        if isinstance(pid, h5py.Dataset):
            try:
                pid_str = "".join([chr(c[0]) for c in pid[:]])
            except:
                pid_str = str(pid[0, 0])
        else:
            pid_str = str(pid)
            
        # Parse image and transpose to get (Height, Width) from HDF5
        image = cjdata['image'][:].T
        
        # Parse tumor border
        border = cjdata['tumorBorder'][0]
        
    return pid_str, label, image, border

def calculate_yolo_bbox(border, img_width, img_height):
    """
    Convert polygon tumorBorder [x1, y1, x2, y2, ...] to YOLO bbox [x_center, y_center, width, height]
    normalized to [0, 1].
    """
    x_coords = border[0::2]
    y_coords = border[1::2]
    
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    # Clip to image boundaries just in case
    x_min = max(0, min(x_min, img_width - 1))
    x_max = max(0, min(x_max, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    y_max = max(0, min(y_max, img_height - 1))
    
    box_w = x_max - x_min
    box_h = y_max - y_min
    x_center = x_min + box_w / 2.0
    y_center = y_min + box_h / 2.0
    
    return x_center / img_width, y_center / img_height, box_w / img_width, box_h / img_height

def main():
    raw_dir = os.path.join("data", "raw")
    images_dir = os.path.join("data", "dataset", "images", "all")
    labels_dir = os.path.join("data", "dataset", "labels", "all")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Get all .mat files except cvind.mat
    mat_files = [f for f in os.listdir(raw_dir) if f.endswith('.mat') and f != "cvind.mat"]
    
    for f in tqdm(mat_files, desc="Generating 2D dataset"):
        path = os.path.join(raw_dir, f)
        base_name = f.split('.')[0]
        try:
            _, label, image, border = process_mat_file(path)

            # Min-max normalize to uint8 and replicate to 3-channel RGB
            img_8u = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_rgb = np.stack((img_8u, img_8u, img_8u), axis=-1)

            h, w = image.shape
            x_c, y_c, bw, bh = calculate_yolo_bbox(border, w, h)

            img_path = os.path.join(images_dir, f"{base_name}.jpg")
            lbl_path = os.path.join(labels_dir, f"{base_name}.txt")

            cv2.imwrite(img_path, img_rgb)

            with open(lbl_path, "w") as lf:
                lf.write(f"{label} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("Dataset preparation complete!")

if __name__ == '__main__':
    main()
