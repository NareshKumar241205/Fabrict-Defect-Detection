import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk


class TextureInspector:
    def __init__(self):
        # -------------------------------
        # Texture parameters (UNCHANGED)
        # -------------------------------
        self.RADIUS = 3
        self.N_POINTS = 8 * self.RADIUS
        self.METHOD = "uniform"

        # -------------------------------
        # Structural decision tuning
        # -------------------------------
        self.STRUCTURE_EDGE_THRESHOLD = 0.55
        self.STRUCTURE_DOMINANCE_RATIO = 0.06

    # ======================================================
    # PREPROCESS (UNCHANGED)
    # ======================================================
    def _preprocess(self, img_buffer):
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)

        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        target_w = 800
        scale = target_w / w

        img_small = cv2.resize(img, (target_w, int(h * scale)))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        return img, img_small, img_gray, scale

    # ======================================================
    # TEXTURE FEATURES (UNCHANGED)
    # ======================================================
    def compute_texture_map(self, img_gray):
        lbp = local_binary_pattern(
            img_gray, self.N_POINTS, self.RADIUS, self.METHOD
        )
        lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
        return lbp_norm.astype(np.uint8)

    def compute_entropy_map(self, lbp_img):
        ent_img = entropy(lbp_img, disk(5))
        return cv2.normalize(
            ent_img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    # ======================================================
    # 🆕 STRUCTURAL VETO (GLOBAL – DECISION ONLY)
    # ======================================================
    def _structural_veto(self, gray):
        # Sobel edges
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        mag_x = np.abs(gx)
        mag_y = np.abs(gy)

        # Projection profiles
        proj_x = np.mean(mag_x, axis=0)
        proj_y = np.mean(mag_y, axis=1)

        proj_x /= (proj_x.max() + 1e-6)
        proj_y /= (proj_y.max() + 1e-6)

        vert_band = np.sum(proj_x > self.STRUCTURE_EDGE_THRESHOLD) / len(proj_x)
        hori_band = np.sum(proj_y > self.STRUCTURE_EDGE_THRESHOLD) / len(proj_y)

        if vert_band > self.STRUCTURE_DOMINANCE_RATIO:
            return "Structural Seam / Missing Warp"

        if hori_band > self.STRUCTURE_DOMINANCE_RATIO:
            return "Structural Seam / Missing Weft"

        return None

    # ======================================================
    # MAIN PIPELINE (ORIGINAL + FINAL DECISION HARDENING)
    # ======================================================
    def detect_defects(self, img_buffer, sensitivity=3.0, min_area=200):
        orig_full, img_small, img_gray, scale = self._preprocess(img_buffer)

        # -------------------------------
        # TEXTURE ANALYSIS (UNCHANGED)
        # -------------------------------
        lbp_map = self.compute_texture_map(img_gray)
        entropy_map = self.compute_entropy_map(lbp_map)

        mean_ent = np.mean(entropy_map)
        std_ent = np.std(entropy_map)

        lower = mean_ent - sensitivity * std_ent
        upper = mean_ent + sensitivity * std_ent

        mask = cv2.bitwise_or(
            cv2.inRange(entropy_map, 0, lower),
            cv2.inRange(entropy_map, upper, 255)
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        defects = []
        output = orig_full.copy()
        bg_mean = np.mean(img_gray)

        # -------------------------------
        # COMPONENT DEFECTS (UNCHANGED)
        # -------------------------------
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            real_area = area / (scale ** 2)

            if real_area < min_area:
                continue

            x_s, y_s, w_s, h_s = stats[i, :4]
            x = int(x_s / scale)
            y = int(y_s / scale)
            w = int(w_s / scale)
            h = int(h_s / scale)

            roi = img_gray[y_s:y_s + h_s, x_s:x_s + w_s]
            roi_mean = np.mean(roi) if roi.size > 0 else bg_mean
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

            if roi_mean < (bg_mean - 25) and aspect_ratio < 2.0:
                label = "Stain / Oil"
                color = (0, 140, 255)
            elif aspect_ratio > 4.0:
                label = "Cut / Tear"
                color = (0, 0, 255)
            else:
                label = "Texture Defect"
                color = (255, 0, 255)

            defects.append({
                "ID": i,
                "Type": label,
                "Area (px)": int(real_area)
            })

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                output, label,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # ======================================================
        # 🔥 FINAL INDUSTRIAL DECISION HARDENING
        # ======================================================
        if len(defects) == 0:
            structure_defect = self._structural_veto(img_gray)
            if structure_defect is not None:
                defects.append({
                    "ID": 0,
                    "Type": structure_defect,
                    "Area (px)": 0
                })

                cv2.putText(
                    output,
                    structure_defect,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

        return orig_full, lbp_map, entropy_map, output, defects


# ======================================================
# EXPORTED INSTANCE
# ======================================================
inspector = TextureInspector()
