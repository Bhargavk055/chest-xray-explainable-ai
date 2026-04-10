import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Path to the dataset
BASE_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, 'data', 'classification', 'chest_xray', 'test')

# Find 2 normal and 2 pneumonia images
normal_files = [os.path.join(DATA_DIR, 'NORMAL', f) for f in os.listdir(os.path.join(DATA_DIR, 'NORMAL')) if f.endswith('.jpeg')][:2]
pneumo_files = [os.path.join(DATA_DIR, 'PNEUMONIA', f) for f in os.listdir(os.path.join(DATA_DIR, 'PNEUMONIA')) if f.endswith('.jpeg')][:2]
files = normal_files + pneumo_files

def segment_lungs_floodfill(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. Threshold the image to separate bright (body) and dark (lungs + background)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # thresh is now: Body=White, Lungs=Black, Background=Black
    
    # 2. Floodfill from the corners to turn the background White
    # Copy the thresholded image.
    im_floodfill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Fill from the 4 corners (assuming background touches the corners)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    cv2.floodFill(im_floodfill, mask, (w-1,0), 255)
    cv2.floodFill(im_floodfill, mask, (0,h-1), 255)
    cv2.floodFill(im_floodfill, mask, (w-1,h-1), 255)
    
    # 3. Now: Body=White, Background=White, Lungs=Black. 
    # Invert the image so Lungs=White!
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # 4. Clean up the lung masks (opening to remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opened = cv2.morphologyEx(im_floodfill_inv, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 5. Extract only the two largest valid contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > (h*w*0.015): # Lungs must be >1.5% of total area
            valid_contours.append(cnt)
            
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
    cv2.drawContours(final_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
    
    return final_mask

fig, axes = plt.subplots(4, 3, figsize=(10, 12))
for i, path in enumerate(files):
    lbl = 'NORMAL' if i < 2 else 'PNEUMO'
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    
    mask = segment_lungs_floodfill(img_rgb)
    masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    axes[i, 0].imshow(img_rgb)
    axes[i, 0].set_title(f'Orig ({lbl})')
    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 2].imshow(masked)
plt.savefig('outputs/test_floodfill.png')
print("Test completed. See outputs/test_floodfill.png")
