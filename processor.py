import cv2
import numpy as np

# --- HELPER: Gabor Filter Generator ---
def build_gabor_filter(ksize, sigma, theta, lam, gamma, psi):
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, psi, ktype=cv2.CV_32F)

# --- FEATURE 1: AUTO-CALIBRATION ---
def auto_calibrate(image_file):
    image_file.seek(0)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity, std_intensity = cv2.meanStdDev(img_gray)
    mean_val = mean_intensity[0][0]
    std_val = std_intensity[0][0]
    
    # Formula: Mean - (3 * StdDev) -> Less aggressive than 4
    suggested_dark_thresh = int(mean_val - (3 * std_val))
    suggested_dark_thresh = max(30, min(suggested_dark_thresh, 230))

    # Texture Calibration
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    img_blur = cv2.GaussianBlur(img_clahe, (5,5), 0)
    
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 4): 
        kern = build_gabor_filter(21, 4.0, theta, 10.0, 0.5, 0)
        filters.append(kern)

    accum_energy = np.zeros_like(img_blur, dtype=np.float32)
    for kern in filters:
        fimg = cv2.filter2D(img_blur, cv2.CV_32F, kern)
        np.maximum(accum_energy, fimg, accum_energy)
    
    accum_energy = cv2.normalize(accum_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tex_mean, tex_std = cv2.meanStdDev(accum_energy)
    
    raw_thresh = tex_mean[0][0] + (4 * tex_std[0][0]) 
    suggested_sensitivity = int(255 - raw_thresh)
    suggested_sensitivity = max(50, min(suggested_sensitivity, 250))

    return suggested_sensitivity, suggested_dark_thresh

# --- FEATURE 2: CLASSIFICATION LOGIC ---
def classify_defect(aspect_ratio, solidity, mean_intensity, dark_thresh):
    if mean_intensity < dark_thresh:
        if mean_intensity < (dark_thresh - 30): return "Hole", (0, 0, 255)
        return "Oil Stain", (0, 165, 255)

    if aspect_ratio > 4.0 or aspect_ratio < 0.20:
        return "Cut / Tear", (0, 0, 255)
    
    if solidity < 0.65:
        return "Snag / Knot", (255, 0, 255)

    return "Surface Defect", (0, 255, 255)

# --- FEATURE 3: MAIN PIPELINE ---
def process_image(image_file, sensitivity_thresh, min_area, dark_thresh):
    image_file.seek(0)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    final_output = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. LIGHTER Preprocessing (Don't blur away the defect!)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    img_blur = cv2.GaussianBlur(img_clahe, (5,5), 0) # Simple blur is safer than Bilateral

    # 2. Texture (Gabor)
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 4): 
        # Reduced kernel size 31->21 to catch smaller textures
        kern = build_gabor_filter(21, 4.0, theta, 10.0, 0.5, 0)
        filters.append(kern)

    accum_energy = np.zeros_like(img_blur, dtype=np.float32)
    for kern in filters:
        fimg = cv2.filter2D(img_blur, cv2.CV_32F, kern)
        np.maximum(accum_energy, fimg, accum_energy) 
    
    accum_energy = cv2.normalize(accum_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_thresh_val = 255 - sensitivity_thresh
    _, mask_texture = cv2.threshold(accum_energy, gabor_thresh_val, 255, cv2.THRESH_BINARY)

    # 3. Intensity (Stains)
    _, mask_intensity = cv2.threshold(img_blur, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    # 4. Combine
    mask_combined = cv2.bitwise_or(mask_texture, mask_intensity)

    # --- CRITICAL FIX: MICRO-CLEANING ---
    # Old: (7,7) kernel -> Deleted small defects
    # New: (3,3) kernel -> Keeps small defects
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Only 1 iteration of opening to remove single-pixel noise
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing to fill gaps inside the defect
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    defect_list = [] 
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # --- DEBUG VISUALIZATION ---
        # If it's too small, draw a BLUE box just so we know the camera saw it
        # This proves "Detection" works, even if "Filtering" removes it
        if area < min_area:
            cv2.rectangle(final_output, (x, y), (x+w, y+h), (255, 0, 0), 1) # Thin Blue Box
            continue

        mask_roi = (labels == i).astype(np.uint8)
        mean_val = cv2.mean(img_blur, mask=mask_roi)[0]
        aspect_ratio = float(w) / h
        
        roi_contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0
        if roi_contours:
            hull_area = cv2.contourArea(cv2.convexHull(roi_contours[0]))
            if hull_area > 0: solidity = area / hull_area

        # Wrinkle Logic
        is_wrinkle = False
        if mean_val > (dark_thresh + 40) and aspect_ratio > 3.0:
            is_wrinkle = True
            
        if is_wrinkle:
            name = "Wrinkle"
            color = (100, 0, 0) # Dark Blue
        else:
            name, color = classify_defect(aspect_ratio, solidity, mean_val, dark_thresh)

        defect_list.append({"ID": i, "Type": name, "Area": area, "Severity": int(255-mean_val)})

        # Draw Final Box
        cv2.rectangle(final_output, (x, y), (x+w, y+h), color, 3)
        
        label = f"{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = y - 10 if y - 10 > 10 else y + h + 20
        cv2.rectangle(final_output, (x, text_y - th - 5), (x + tw, text_y + 5), color, -1)
        cv2.putText(final_output, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img, mask_clean, final_output, defect_list