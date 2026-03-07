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
from typing import List, Dict, Any, BinaryIO, Tuple

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
    # Pre-classifier
    # ──────────────────────────────────────────
    def _pre_classify(self, img_gray: np.ndarray) -> Dict[str, bool]:
        """Determine whether a seam region and/or fabric body are present.

        FIX: gradient threshold lowered from 0.6 to a tunable value
        so real seams are not rejected.
        """
        h, w = img_gray.shape[:2]
        h_proj = np.mean(img_gray, axis=1)

        p_min, p_max = float(np.min(h_proj)), float(np.max(h_proj))
        if p_max - p_min < 10:
            return {"has_seam": False, "has_fabric_body": True}

        h_proj_norm = (h_proj - p_min) / (p_max - p_min)
        grad = np.abs(np.gradient(h_proj_norm))

        # FIX: lower gradient threshold — the old code used max(0.4, thresh*2)
        # which was 0.6 with the default 0.3, causing real seams to be ignored.
        edge_thresh = max(0.15, self._seam_thresh)
        strong_edges = np.where(grad > edge_thresh)[0]

        has_seam = False
        if len(strong_edges) >= 2:
            for i in range(len(strong_edges) - 1):
                band_width = strong_edges[i + 1] - strong_edges[i]
                if 10 < band_width < h * 0.40:
                    band_slice = h_proj_norm[strong_edges[i] : strong_edges[i + 1]]
                    if len(band_slice) > 3 and np.std(band_slice) < 0.25:
                        has_seam = True
                        break

        return {"has_seam": has_seam, "has_fabric_body": True}

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
        self, img_buffer: BinaryIO, sensitivity: float  # noqa: ARG002
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Group II — Stitch Quality (Seam Inspector)."""
        defects: List[Dict[str, Any]] = []
        viz_maps: Dict[str, Any] = {}

        try:
            img_buffer.seek(0)
            buf = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            _, _, _, seam_output, seam_defs = self._seam.detect_defects(buf)
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

        # Pre-classify
        pre_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        img_pre = cv2.imdecode(pre_bytes, cv2.IMREAD_GRAYSCALE)
        if img_pre is None:
            raise ValueError("Could not decode image file")

        region_info = self._pre_classify(img_pre)

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
            g2_defects, g2_viz = self._route_group_ii(pipeline_buffer, sensitivity)
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

        # ── Stage 2: Confidence gate ──────────────────────────────────────────
        # Drop anything the pipeline isn't reasonably confident about.
        # Low-confidence detections (10-30%) are almost always texture noise
        # on plain fabric being mistaken for a defect.
        MIN_CONFIDENCE = 40  # % — tune down if real defects are being missed

        def _conf_val(d: Dict[str, Any]) -> int:
            c = d.get("Confidence", "0%")
            try:
                return int(str(c).replace("%", "").strip())
            except (ValueError, TypeError):
                return 0

        all_defects = [d for d in all_defects if _conf_val(d) >= MIN_CONFIDENCE]

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
