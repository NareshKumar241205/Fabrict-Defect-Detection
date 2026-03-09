"""
Unified Defect Processor — Intelligent Routing Pipeline
========================================================
Routes fabric images to the correct detection engine based on image content.

Architecture:
    Input Image → Shadow Removal → Pre-classifier → Group Router
                → Detection Engines → Sub-classifier → NMS → Unified Result

Groups:
    Group I  (Fabric Structure): Spectral + Texture + Edge engines
    Group II (Stitch Quality):   Projection + Regression + Laplacian engines

Output schema per defect:
    {ID, Group, Engine, Inspector, Type, Area, Confidence, Solidity, bbox_*}
"""

import cv2
import numpy as np
import logging
from io import BytesIO
from typing import List, Dict, Any, BinaryIO, Tuple, Optional

from config import DEFECT_TYPES, UNIFIED_SETTINGS
from inspectors.spectral_inspector import SpectralInspector
from inspectors.edge_inspector import EdgeInspector
from inspectors.texture_inspector import TextureInspector
from inspectors.seam_inspector import SeamInspector
from inspectors.reference_inspector import ReferenceInspector

logger = logging.getLogger(__name__)


class UnifiedProcessor:
    """Intelligent routing processor for fabric defect detection."""

    DEFECT_TAXONOMY = DEFECT_TYPES

    def __init__(self):
        self._spectral = SpectralInspector()
        self._edge = EdgeInspector()
        self._texture = TextureInspector()
        self._seam = SeamInspector()
        self._reference = ReferenceInspector()
        self._cfg = UNIFIED_SETTINGS
        self._seam_thresh = self._cfg.get("SEAM_DETECTION_THRESH", 0.3)

    # ──────────────────────────────────────────
    # Shadow / Illumination removal
    # ──────────────────────────────────────────
    def _remove_shadows(self, img_bytes: bytes) -> bytes:
        """Morphological background estimation in LAB L-channel
        (avoids the colour-shift issue of the old HSV approach).
        """
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return img_bytes

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]

        k_size = max(31, (img.shape[1] // 20) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        bg = cv2.morphologyEx(l_chan, cv2.MORPH_DILATE, kernel)
        bg = cv2.GaussianBlur(bg, (k_size, k_size), 0)
        bg[bg == 0] = 1

        normalized = (l_chan.astype(np.float32) / bg.astype(np.float32)) * 255
        lab[:, :, 0] = np.clip(normalized, 0, 255).astype(np.uint8)

        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        ok, encoded = cv2.imencode(".png", corrected)
        return encoded.tobytes() if ok else img_bytes

    # ──────────────────────────────────────────
    # Deskew — Canny + Hough median-angle correction
    # (Extracted from SeamInspector for pipeline-wide use)
    # ──────────────────────────────────────────
    @staticmethod
    def _deskew(img_gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Correct slight fabric skew using Canny edges + Hough lines.

        Returns the rotated grayscale image and the applied angle (degrees).
        """
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return img_gray, 0.0

        angles: List[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if 0.5 < abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return img_gray, 0.0

        median_angle = float(np.median(angles))
        h, w = img_gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rot_img = cv2.warpAffine(img_gray, M, (w, h))
        return rot_img, median_angle

    @staticmethod
    def _deskew_color(img_bgr: np.ndarray, angle: float) -> np.ndarray:
        """Apply the same rotation to a colour image so bboxes stay aligned."""
        if abs(angle) < 0.01:
            return img_bgr
        h, w = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h))

    # ──────────────────────────────────────────
    # Pre-classifier
    # ──────────────────────────────────────────
    def _pre_classify(self, img_gray: np.ndarray) -> Dict[str, bool]:
        """Determine whether a seam region and/or fabric body are present.

        Strategy:
          1. Canny + HoughLinesP — look for a dominant long line (any orientation).
             A seam is a strong, roughly continuous line crossing a large portion
             of the image.  This works regardless of seam orientation.
          2. Fallback: check both horizontal AND vertical mean-intensity projections
             for sharp gradient spikes that indicate a seam band.

        Returns:
            has_seam: True if a seam / stitch line is likely present
            has_fabric_body: always True (fabric is always present)
            seam_orientation: "horizontal", "vertical", or None
        """
        h, w = img_gray.shape[:2]

        # ── Method 1: Canny + HoughLinesP ──
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        min_dim = min(h, w)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80,
            minLineLength=min_dim * 0.30,   # line must span ≥30 % of image
            maxLineGap=15,
        )

        if lines is not None and len(lines) > 0:
            # Classify each line as horizontal-ish or vertical-ish
            # AND record line mid-points for clustering check
            horiz_mids: list = []   # mid-Y of horizontal lines
            vert_mids: list = []    # mid-X of vertical lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < min_dim * 0.25:
                    continue
                if angle < 20 or angle > 160:       # tighter horizontal band
                    horiz_mids.append((y1 + y2) / 2.0)
                elif 70 < angle < 110:               # tighter vertical band
                    vert_mids.append((x1 + x2) / 2.0)

            def _cluster_count(mids: list, span: int, min_votes: int = 3) -> int:
                """Return max number of lines that cluster within 25% of span."""
                if len(mids) < min_votes:
                    return 0
                mids_s = sorted(mids)
                band = span * 0.25
                best = 0
                for i in range(len(mids_s)):
                    count = sum(1 for m in mids_s[i:] if m - mids_s[i] <= band)
                    best = max(best, count)
                return best

            h_count = _cluster_count(horiz_mids, h)
            v_count = _cluster_count(vert_mids, w)

            if h_count >= 3 or v_count >= 3:
                orientation = "horizontal" if h_count >= v_count else "vertical"
                return {"has_seam": True, "has_fabric_body": True,
                        "seam_orientation": orientation}

        # ── Method 2: Projection gradient (both axes) ──
        def _check_projection(proj: np.ndarray, dim: int) -> bool:
            p_min, p_max = float(np.min(proj)), float(np.max(proj))
            if p_max - p_min < 10:
                return False
            proj_norm = (proj - p_min) / (p_max - p_min)
            grad = np.abs(np.gradient(proj_norm))
            edge_thresh = max(0.12, self._seam_thresh)
            strong = np.where(grad > edge_thresh)[0]
            if len(strong) >= 2:
                for i in range(len(strong) - 1):
                    band = strong[i + 1] - strong[i]
                    if 10 < band < dim * 0.40:
                        band_slice = proj_norm[strong[i]:strong[i + 1]]
                        if len(band_slice) > 3 and np.std(band_slice) < 0.25:
                            return True
            return False

        h_proj = np.mean(img_gray, axis=1)  # one value per row → horizontal seam
        v_proj = np.mean(img_gray, axis=0)  # one value per col → vertical seam

        has_h = _check_projection(h_proj, h)
        has_v = _check_projection(v_proj, w)

        if has_h or has_v:
            orientation = "horizontal" if has_h else "vertical"
            return {"has_seam": True, "has_fabric_body": True,
                    "seam_orientation": orientation}

        return {"has_seam": False, "has_fabric_body": True,
                "seam_orientation": None}

    # ──────────────────────────────────────────
    # Sub-classifiers
    # ──────────────────────────────────────────
    def _sub_classify_spectral(self, defect: Dict[str, Any]) -> str:
        """Refine a spectral / texture detection into the 10-type taxonomy.

        FIX: the old code short-circuited on "Anomaly" / "Weave" keywords
        and returned "Texture Defect", bypassing the taxonomy.  Removed.
        """
        solidity = 0.0
        sol_raw = defect.get("Solidity", "0")
        try:
            solidity = float(str(sol_raw).replace("%", ""))
        except (ValueError, TypeError):
            pass

        area = defect.get("Area (px)", 0)
        bw = defect.get("bbox_w", 1)
        bh = defect.get("bbox_h", 1)
        aspect = bw / max(bh, 1)

        sol_min = self._cfg.get("OIL_STAIN_SOLIDITY_MIN", 0.85)

        if solidity > sol_min and 0.4 < aspect < 2.5:
            return "Oil Stain"

        if aspect > 3.0 or aspect < 0.33:
            conf_str = defect.get("Confidence", "0%")
            try:
                conf = int(str(conf_str).replace("%", ""))
            except (ValueError, TypeError):
                conf = 0
            return "Missing Thread" if conf >= 30 else "Slub"

        return "Slub"

    def _sub_classify_edge(self, defect: Dict[str, Any]) -> str:
        """Refine an edge detection into Hole / Tear / Snag.

        FIX: removed short-circuit on "Wrinkle" / "Weave" that returned
        "Texture Defect" and prevented proper classification.
        """
        area = defect.get("Area (px)", 0)
        bw = defect.get("bbox_w", 1)
        bh = defect.get("bbox_h", 1)
        aspect = bw / max(bh, 1)

        tear_ar = self._cfg.get("TEAR_ASPECT_RATIO_MIN", 3.0)
        snag_max = self._cfg.get("SNAG_AREA_MAX", 800)
        hole_min = self._cfg.get("HOLE_AREA_MIN", 1000)

        if aspect > tear_ar or aspect < (1.0 / tear_ar):
            return "Tear"
        if area < snag_max:
            return "Snag"
        if area >= hole_min:
            return "Hole"
        return "Hole"

    # ──────────────────────────────────────────
    # Internal NMS (deduplication)
    # ──────────────────────────────────────────
    @staticmethod
    def _nms(defects: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
        """Non-Maximum Suppression across engines to remove duplicates.

        FIX: the old code had NO internal NMS — it relied entirely on
        ``app.py`` to do it.  Now deduplicated inside the processor.
        """
        if len(defects) <= 1:
            return defects

        def _conf(d: Dict[str, Any]) -> int:
            # Prefer the numeric Severity field; fall back to parsing Confidence string
            if "Severity" in d:
                return int(d["Severity"])
            c = d.get("Confidence", "0%")
            try:
                return int(str(c).replace("%", "").strip())
            except (ValueError, TypeError):
                return 0

        def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
            ax1 = a.get("bbox_x", 0)
            ay1 = a.get("bbox_y", 0)
            ax2 = ax1 + a.get("bbox_w", 0)
            ay2 = ay1 + a.get("bbox_h", 0)
            bx1 = b.get("bbox_x", 0)
            by1 = b.get("bbox_y", 0)
            bx2 = bx1 + b.get("bbox_w", 0)
            by2 = by1 + b.get("bbox_h", 0)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
            return inter / union

        sorted_defs = sorted(defects, key=_conf, reverse=True)
        keep: List[Dict[str, Any]] = []
        for d in sorted_defs:
            if not any(_iou(d, k) >= iou_thresh for k in keep):
                keep.append(d)
        return keep

    # ──────────────────────────────────────────
    # Photometric & Gradient Rule Engine (FP Filter)
    # ──────────────────────────────────────────
    def _validate_defect_geometry(
        self, img_gray: np.ndarray, defect: Dict[str, Any]
    ) -> bool:
        """Classical photometric + gradient validation of a bounding box.

        Compares the raw grayscale pixels *inside* the defect bbox to a
        slightly expanded local neighborhood.  If the defect does not stand
        out from its surroundings (low contrast or no sharp edge), it is
        likely a shadow, fold, or lighting artefact — return False to drop.

        Returns True  → keep the defect.
                False → drop (false positive).
        """
        h_img, w_img = img_gray.shape[:2]
        bx = defect.get("bbox_x", 0)
        by = defect.get("bbox_y", 0)
        bw = defect.get("bbox_w", 1)
        bh = defect.get("bbox_h", 1)

        # Clamp inner ROI to image bounds
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w_img, bx + bw)
        y2 = min(h_img, by + bh)
        if x2 <= x1 or y2 <= y1:
            return False

        inner = img_gray[y1:y2, x1:x2]
        if inner.size == 0:
            return False

        # Expand bbox by 50 % on each side for local neighbourhood
        expand_x = max(10, bw // 2)
        expand_y = max(10, bh // 2)
        nx1 = max(0, x1 - expand_x)
        ny1 = max(0, y1 - expand_y)
        nx2 = min(w_img, x2 + expand_x)
        ny2 = min(h_img, y2 + expand_y)
        neighbourhood = img_gray[ny1:ny2, nx1:nx2]
        if neighbourhood.size == 0:
            return False

        # ── Test 1: Local Contrast Ratio ──
        # NOTE: Texture-engine defects (NCC template matching) detect *pattern*
        # anomalies, not intensity anomalies.  A Slub may have the same mean
        # brightness as its surroundings yet have very different texture.
        # Skip the contrast gate for those; they already passed texture
        # deviation thresholding inside the engine.
        engine = defect.get("Engine", "")
        is_texture_engine = "Texture" in engine

        inner_mean = float(np.mean(inner))
        neigh_mean = float(np.mean(neighbourhood))
        if neigh_mean < 1.0:
            neigh_mean = 1.0
        contrast_ratio = abs(inner_mean - neigh_mean) / neigh_mean

        MIN_CONTRAST = self._cfg.get("FP_MIN_CONTRAST", 0.05)
        if (not is_texture_engine) and contrast_ratio < MIN_CONTRAST:
            return False  # barely different from surroundings — shadow / noise

        # ── Test 2: Edge Gradient Strength (Sobel) ──
        sobel_x = cv2.Sobel(inner, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(inner, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mean_gradient = float(np.mean(gradient_mag))

        MIN_GRADIENT = self._cfg.get("FP_MIN_GRADIENT", 8.0)
        if mean_gradient < MIN_GRADIENT:
            return False  # soft, diffuse transition — likely a fold or shadow

        return True

    # ──────────────────────────────────────────
    # Group routers
    # ──────────────────────────────────────────
    def _route_group_i(
        self, img_buffer: BinaryIO, sensitivity: float
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Group I — Fabric Structure (Spectral + Texture + Edge)."""
        defects: List[Dict[str, Any]] = []
        viz_maps: Dict[str, Any] = {}

        # Engine A: Spectral (Gabor)
        try:
            img_buffer.seek(0)
            buf = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            spec_defs, sal_map = self._spectral.process(buf, sensitivity=sensitivity)
            viz_maps["saliency"] = sal_map
            defects.extend(spec_defs)
        except Exception as e:
            logger.warning("Spectral (Gabor) engine failed: %s", e)

        # Engine B: Texture (SSIM)
        try:
            img_buffer.seek(0)
            buf = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            tex_defs, ent_map = self._texture.process(buf, sensitivity=sensitivity)
            viz_maps["entropy"] = ent_map
            defects.extend(tex_defs)
        except Exception as e:
            logger.warning("Texture (SSIM) engine failed: %s", e)

        # Engine C: Edge (Subtraction)
        try:
            img_buffer.seek(0)
            buf = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            edge_defs, anomaly_hm = self._edge.detect_defects(buf, sensitivity=sensitivity)
            viz_maps["anomaly_heatmap"] = anomaly_hm
            defects.extend(edge_defs)
        except Exception as e:
            logger.warning("Edge (Subtraction) engine failed: %s", e)

        return defects, viz_maps

    def _route_group_ii(
        self, img_buffer: BinaryIO, sensitivity: float,  # noqa: ARG002
        seam_orientation: str = "horizontal",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Group II — Stitch Quality (Seam Inspector)."""
        defects: List[Dict[str, Any]] = []
        viz_maps: Dict[str, Any] = {}

        try:
            img_buffer.seek(0)
            buf = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            _, _, _, seam_output, seam_defs = self._seam.detect_defects(
                buf, orientation=seam_orientation,
            )
            viz_maps["seam_output"] = seam_output
            for d in seam_defs:
                d["Engine"] = f"Seam ({d.get('Type', 'Unknown')})"
                d["Group"] = "Stitch Quality"
                d["Inspector"] = "Seam"
            defects.extend(seam_defs)
        except Exception as e:
            logger.warning("Seam engine failed: %s", e)

        return defects, viz_maps

    # ──────────────────────────────────────────
    # Reference-Based Inspection (Golden Image + SSIM)
    # ──────────────────────────────────────────
    def _route_reference(
        self, img_buffer: BinaryIO, ref_buffer: BinaryIO, sensitivity: float
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Reference Compare — align test to golden image, SSIM diff."""
        defects: List[Dict[str, Any]] = []
        viz_maps: Dict[str, Any] = {}

        try:
            img_buffer.seek(0)
            buf_test = BytesIO(img_buffer.read())
            img_buffer.seek(0)

            ref_buffer.seek(0)
            buf_ref = BytesIO(ref_buffer.read())
            ref_buffer.seek(0)

            _, result_img, ssim_heatmap, ref_defs = self._reference.detect_defects(
                buf_test, buf_ref, sensitivity=sensitivity
            )
            viz_maps["ssim_heatmap"] = ssim_heatmap
            viz_maps["reference_result"] = result_img
            for d in ref_defs:
                d["Engine"] = "Reference (SSIM)"
                d["Group"] = "Fabric Structure"
                d["Inspector"] = "Reference"
            defects.extend(ref_defs)
        except Exception as e:
            logger.warning("Reference engine failed: %s", e)

        return defects, viz_maps

    # ──────────────────────────────────────────
    # Main API
    # ──────────────────────────────────────────
    def process(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.5,
        mode: str = "full",
        remove_shadows: bool = False,
        ref_buffer: BinaryIO = None,
    ) -> Dict[str, Any]:
        """Run the full Unified Pipeline.

        Args:
            ref_buffer: Optional golden-image buffer for Reference Compare mode.

        Returns dict with keys: defects, group_i_defects, group_ii_defects,
        viz_maps, routing_info, summary.
        """
        all_defects: List[Dict[str, Any]] = []
        all_viz_maps: Dict[str, Any] = {}
        engines_used: List[str] = []

        img_buffer.seek(0)
        raw_bytes = img_buffer.read()
        if remove_shadows:
            raw_bytes = self._remove_shadows(raw_bytes)

        pipeline_buffer = BytesIO(raw_bytes)

        # ── Deskew for pre-classification ONLY ──
        # Deskew corrects horizontal projection for the seam detector gate,
        # but passing the rotated image to detection engines creates black
        # border artifacts that look like dark defects.  So deskew is
        # applied to a COPY used solely by _pre_classify; all engines
        # and the geometry validator receive the original image.
        pre_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        img_pre_gray = cv2.imdecode(pre_bytes, cv2.IMREAD_GRAYSCALE)
        if img_pre_gray is None:
            raise ValueError("Could not decode image file")

        img_deskewed, _deskew_angle = self._deskew(img_pre_gray)

        # Pre-classify on the deskewed grayscale (better horizontal projection)
        region_info = self._pre_classify(img_deskewed)

        # Group I (fabric structure) always runs.
        # Group II (seam/stitch) only runs if the pre-classifier detected a seam.
        # In "full" mode we still respect the seam gate — running on plain fabric
        # causes the knit loop pattern to be misclassified as Crooked Stitch.
        run_group_i = mode in ("full", "structure_only") or region_info["has_fabric_body"]
        run_group_ii = (mode == "seam_only") or region_info["has_seam"]

        if run_group_i:
            engines_used.append("Group I: Fabric Structure")
            g1_defects, g1_viz = self._route_group_i(pipeline_buffer, sensitivity)
            all_defects.extend(g1_defects)
            all_viz_maps.update(g1_viz)

        if run_group_ii:
            engines_used.append("Group II: Stitch Quality")
            orientation = region_info.get("seam_orientation", "horizontal")
            g2_defects, g2_viz = self._route_group_ii(
                pipeline_buffer, sensitivity,
                seam_orientation=orientation or "horizontal",
            )
            all_defects.extend(g2_defects)
            all_viz_maps.update(g2_viz)

        # Reference Compare (SSIM Golden Image) if reference provided
        if ref_buffer is not None:
            engines_used.append("Reference Compare (SSIM)")
            ref_defects, ref_viz = self._route_reference(
                pipeline_buffer, ref_buffer, sensitivity
            )
            all_defects.extend(ref_defects)
            all_viz_maps.update(ref_viz)

        # ── Stage 1: Tighter NMS (0.35 instead of 0.50) ─────────────────────
        # A lower IoU threshold means boxes that partially overlap (same defect
        # seen by two inspectors) get merged into one.  Previously at 0.5 they
        # were kept as two separate detections, doubling the count.
        all_defects = self._nms(all_defects, iou_thresh=0.35)

        # ── Stage 2: Severity gate ────────────────────────────────────────────
        # Drop anything the pipeline isn't reasonably confident about.
        # Uses the multi-metric Severity score (0-100) when available.
        MIN_SEVERITY = 30  # severity score — tune down if real defects are missed

        def _sev_val(d: Dict[str, Any]) -> int:
            if "Severity" in d:
                return int(d["Severity"])
            c = d.get("Confidence", "0%")
            try:
                return int(str(c).replace("%", "").strip())
            except (ValueError, TypeError):
                return 0

        all_defects = [d for d in all_defects if _sev_val(d) >= MIN_SEVERITY]

        # ── Stage 2b: Oversized bbox filter ──────────────────────────────────
        # Drop any bounding box that covers more than 20% of the image.
        # Real defects are localised; a box this large is a false positive.
        # Seam defects (like Crooked Stitch) can span the full seam length
        # so we use a higher threshold (50%) for stitch-quality defects.
        # Crooked Stitch and Pucker naturally span the full seam width, so
        # we use an aspect-ratio guard instead of pure area for them.
        img_h, img_w = img_pre_gray.shape[:2]
        img_area = max(img_h * img_w, 1)
        MAX_FABRIC_RATIO = self._cfg.get("MAX_BOX_AREA_RATIO", 0.22)
        MAX_SEAM_RATIO = 0.50
        # Line-like stitch defects are exempt from area filter; they're wide
        # but narrow.  Use aspect ratio instead: reject only if BOTH area is
        # huge AND the box is roughly square (not line-like).
        LINE_STITCH_TYPES = {"Crooked Stitch", "Pucker", "Run-off Stitch"}

        def _box_ok(d: Dict[str, Any]) -> bool:
            bw = d.get("bbox_w", 0)
            bh = d.get("bbox_h", 0)
            ratio = (bw * bh) / img_area
            if d.get("Type") in LINE_STITCH_TYPES:
                # These span a seam line — wide & narrow is normal.
                # Only reject if >70% of the image AND roughly square.
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                return ratio < 0.70 or aspect > 3.0
            if d.get("Group") == "Stitch Quality":
                return ratio < MAX_SEAM_RATIO
            return ratio < MAX_FABRIC_RATIO

        all_defects = [d for d in all_defects if _box_ok(d)]

        # ── Stage 3: Photometric & Gradient Rule Engine (Task 2) ─────────────
        # Validate each surviving bbox against the raw grayscale.
        # Drop proposals that lack local contrast or sharp edge gradients.
        # Seam (stitch quality) defects are exempt — they detect path /
        # spacing anomalies, not intensity anomalies.
        all_defects = [
            d for d in all_defects
            if d.get("Group") == "Stitch Quality"
            or self._validate_defect_geometry(img_pre_gray, d)
        ]

        # Re-number IDs
        for i, d in enumerate(all_defects, 1):
            d["ID"] = i

        group_i = [d for d in all_defects if d.get("Group") == "Fabric Structure"]
        group_ii = [d for d in all_defects if d.get("Group") == "Stitch Quality"]

        total = len(all_defects)
        summary = {
            "total_defects": total,
            "group_i_count": len(group_i),
            "group_ii_count": len(group_ii),
            "verdict": "PASS" if total == 0 else "FAIL",
            "defect_types_found": sorted(set(d.get("Type", "Unknown") for d in all_defects)),
        }

        return {
            "defects": all_defects,
            "group_i_defects": group_i,
            "group_ii_defects": group_ii,
            "viz_maps": all_viz_maps,
            "routing_info": {
                "pre_classification": region_info,
                "engines_used": engines_used,
                "mode": mode,
            },
            "summary": summary,
        }

    def get_taxonomy(self) -> Dict[str, Any]:
        return self.DEFECT_TAXONOMY


# Module instance
unified_processor = UnifiedProcessor()
