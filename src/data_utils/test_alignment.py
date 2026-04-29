import os
import h5py
import numpy as np
import cv2

def test_alignment():
    mat_path = "data/raw/1.mat"
    with h5py.File(mat_path, 'r') as f_in:
        cjdata = f_in['cjdata']
        # Read image
        img = cjdata['image'][:]
        # Normalise to 8-bit
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        border = cjdata['tumorBorder'][0]
        x_coords = border[0::2]
        y_coords = border[1::2]
        
        # Draw the points on the image to verify alignment
        img_color = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
        for x, y in zip(x_coords, y_coords):
            # Try drawing it without transposing first
            cv2.circle(img_color, (int(x), int(y)), 2, (0, 0, 255), -1)
            
        cv2.imwrite("test_alignment.png", img_color)
        
        # Test transposed
        img_t = img.T
        img_t_8u = cv2.normalize(img_t, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_t_color = cv2.cvtColor(img_t_8u, cv2.COLOR_GRAY2BGR)
        for x, y in zip(x_coords, y_coords):
            cv2.circle(img_t_color, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.imwrite("test_alignment_transposed.png", img_t_color)

if __name__ == '__main__':
    test_alignment()
