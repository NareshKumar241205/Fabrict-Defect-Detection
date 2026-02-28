"""
Unified Defect Processor — Intelligent Routing Pipeline
========================================================
Routes fabric images to the correct detection engine based on defect type.

Architecture:
    Input Image → Pre-classifier → Group Router → Detection Engines → Sub-classifier → Unified Result

Groups:
    Group I  (Fabric Structure): FFT + Canny/Morphology
    Group II (Stitch Quality):   Projection + Regression + Laplacian

Output Schema:
    {group, defect_type, confidence, engine_used, bbox, area, ...}
"""

import cv2
import numpy as np
import logging
from io import BytesIO
from typing import List, Dict, Any, BinaryIO, Optional, Tuple

from config import DEFECT_TYPES, UNIFIED_SETTINGS
from inspectors.spectral_inspector import SpectralInspector
from inspectors.edge_inspector import EdgeInspector
from inspectors.texture_inspector import TextureInspector
from inspectors.seam_inspector import SeamInspector

logger = logging.getLogger(__name__)


class UnifiedProcessor:
    """
    Intelligent routing processor for fabric defect detection.
    
    Instead of running all engines on every image, it:
    1. Pre-classifies the image (fabric body vs seam region)
    2. Routes to the appropriate detection engines
    3. Sub-classifies detected anomalies into the 10-type taxonomy
    4. Returns a unified result schema
    """

    DEFECT_TAXONOMY = DEFECT_TYPES

    def __init__(self):
        # Initialize all engines (lazy — they're lightweight)
        self._spectral = SpectralInspector()
        self._edge = EdgeInspector()
        self._texture = TextureInspector()
        self._seam = SeamInspector()
        
        # Unified settings
        self._cfg = UNIFIED_SETTINGS
        self._seam_thresh = self._cfg.get("SEAM_DETECTION_THRESH", 0.3)

    # ──────────────────────────────────────────────
    # IMAGE PRE-PROCESSING (SHADOW / ILLUMINATION)
    # ──────────────────────────────────────────────
    def _remove_shadows(self, img_bytes: bytes) -> bytes:
        """
        Removes uneven illumination (shadows) via morphological background estimation.
        Decodes the image, estimates the bright fabric background, normalizes,
        and re-encodes back to bytes so inspectors run unmodified.
        """
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return img_bytes
            
        # Convert to HSV to process only the V (brightness) channel without shifting colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        
        # Estimate background illumination:
        # 1. Morphological dilation to wipe out dark defects (threads, holes) and keep the background level
        # A large kernel ensures large defects are erased from background estimation
        # We scale kernel size based on image width to remain robust
        k_size = max(31, (img.shape[1] // 20) | 1) # Must be odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        bg = cv2.morphologyEx(v, cv2.MORPH_DILATE, kernel)
        
        # 2. Strong Gaussian blur to smooth the illumination mask into a perfect gradient/shadow map
        bg = cv2.GaussianBlur(bg, (k_size, k_size), 0)
        
        # Avoid division by zero
        bg[bg == 0] = 1
        
        # Normalize original brightness by estimated background shadow
        normalized = (v.astype(np.float32) / bg.astype(np.float32)) * 255
        hsv[:,:,2] = np.clip(normalized, 0, 255).astype(np.uint8)
        
        img_corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Encode back to PNG buffer
        success, encoded = cv2.imencode('.png', img_corrected)
        if success:
            return encoded.tobytes()
        return img_bytes

    # ──────────────────────────────────────────────
    # PRE-CLASSIFIER
    # ──────────────────────────────────────────────
    def _pre_classify(self, img_gray: np.ndarray) -> Dict[str, bool]:
        """
        Lightweight pre-classifier to determine region types present.
        
        Returns dict: {"has_seam": bool, "has_fabric_body": bool}
        
        Logic:
        - Seam: Detected via horizontal projection profiling.
          If there's a strong, narrow horizontal band of high/low intensity → seam.
        - Fabric body: Always True if not exclusively a seam strip.
        """
        h, w = img_gray.shape[:2]
        
        # Horizontal projection: sum intensities per row
        h_proj = np.mean(img_gray, axis=1)
        
        # Normalize projection
        p_min, p_max = np.min(h_proj), np.max(h_proj)
        if p_max - p_min < 10:
            return {"has_seam": False, "has_fabric_body": True}
        
        h_proj_norm = (h_proj - p_min) / (p_max - p_min)
        
        # Look for a narrow band (< 20% of image height) with distinctly 
        # different intensity than surroundings
        # A seam creates a sharp peak or valley in the projection
        grad = np.abs(np.gradient(h_proj_norm))
        
        # Find strong gradient transitions (seam edges)
        # We increase the required strength threshold dramatically: a real seam
        # stands out heavily compared to the typical background weave.
        strong_edges = np.where(grad > max(0.4, self._seam_thresh * 2))[0]
        
        has_seam = False
        if len(strong_edges) >= 2:
            # Check if there are paired edges close together (forming a band)
            for i in range(len(strong_edges) - 1):
                band_width = strong_edges[i + 1] - strong_edges[i]
                if 15 < band_width < h * 0.35:  # Between 15px and 35% of height
                    # STRICT CHECK: A real seam band has relatively uniform intensity inside it
                    # compared to the sharp edges bounding it.
                    start_idx = strong_edges[i]
                    end_idx = strong_edges[i+1]
                    band_slice = h_proj_norm[start_idx:end_idx]
                    
                    if len(band_slice) > 5:
                        band_std = np.std(band_slice)
                        if band_std < 0.2: # Must be relatively uniform inside the band
                            has_seam = True
                            break
        
        # Fabric body is present unless the image is entirely a zoomed seam
        has_fabric_body = True
        
        return {"has_seam": has_seam, "has_fabric_body": has_fabric_body}

    # ──────────────────────────────────────────────
    # SUB-CLASSIFIERS
    # ──────────────────────────────────────────────
    def _sub_classify_spectral(self, defect: Dict[str, Any]) -> str:
        """
        Refine a spectral/texture anomaly into:
        - Missing Thread: Linear anomaly (high aspect ratio, low solidity)
        - Slub: Small, blob-like (moderate area, moderate solidity)
        - Oil Stain: Round/solid (high solidity, moderate-large area)
        """
        orig_type = defect.get("Type", "")
        # Prevent forcing everything into Structural if it's already a known texture issue
        if "Anomaly" in orig_type or "Weave" in orig_type:
            return "Texture Defect"

        solidity = float(defect.get("Solidity", "0").replace("%", "")) if isinstance(
            defect.get("Solidity"), str) else defect.get("Solidity", 0)
        area = defect.get("Area (px)", 0)
        
        bw = defect.get("bbox_w", 1)
        bh = defect.get("bbox_h", 1)
        aspect = bw / max(bh, 1)
        
        sol_min = self._cfg.get("OIL_STAIN_SOLIDITY_MIN", 0.85)
        
        # High solidity + moderate/large area → Oil Stain
        if solidity > sol_min:
            return "Oil Stain"
        
        # Very elongated → Missing Thread (linear gap in weave)
        if aspect > 3.0 or aspect < 0.33:
            # Check confidence: if it's low confidence, it's just rough weave, not a missing thread
            conf_str = defect.get("Confidence", "0%")
            conf = int(str(conf_str).replace("%", "")) if isinstance(conf_str, str) else conf_str
            if conf < 40:
                return "Rough Weave"
            return "Missing Thread"
        
        # Default: Slub (thick/uneven yarn)
        return "Slub"

    def _sub_classify_edge(self, defect: Dict[str, Any]) -> str:
        """
        Refine an edge/structural anomaly into:
        - Hole: Large area, moderate solidity (complete perforation)
        - Tear: Elongated shape (ripped along a line)
        - Snag: Small area (pulled loop)
        - Texture Defect: Natural fold/wrinkle misclassified by default
        """
        orig_type = defect.get("Type", "")
        # If the original engine called it a Wrinkle or Weave Irregularity,
        # it is a Surface defect, not a Structural Tear/Snag!
        if "Wrinkle" in orig_type or "Weave" in orig_type:
            return "Texture Defect"

        area = defect.get("Area (px)", 0)
        bw = defect.get("bbox_w", 1)
        bh = defect.get("bbox_h", 1)
        aspect = bw / max(bh, 1)
        
        tear_ar = self._cfg.get("TEAR_ASPECT_RATIO_MIN", 3.0)
        snag_max = self._cfg.get("SNAG_AREA_MAX", 800)
        hole_min = self._cfg.get("HOLE_AREA_MIN", 1000)
        
        # Very elongated → Tear
        if aspect > tear_ar or aspect < (1.0 / tear_ar):
            return "Tear"
        
        # Small area → Snag
        if area < snag_max:
            return "Snag"
        
        # Large area → Hole
        if area >= hole_min:
            return "Hole"
        
        # Medium area, not elongated → default Hole
        return "Hole"

    # ──────────────────────────────────────────────
    # GROUP ROUTERS
    # ──────────────────────────────────────────────
    def _route_group_i(self, img_buffer: BinaryIO, sensitivity: float
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
        """
        Group I: Fabric Structure defects.
        Runs Spectral (FFT) + Edge (Canny/Morphology) engines.
        Sub-classifies results into the 6 Group I types.
        """
        defects = []
        viz_maps = {}
        
        # Engine A: Spectral Residual (FFT) → Missing Thread, Slub, Oil Stain
        try:
            img_buffer.seek(0)
            buf_copy = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            
            _, _, sal_map, spec_defs = self._spectral.detect_defects(
                buf_copy, sensitivity=sensitivity
            )
            viz_maps["saliency"] = sal_map
            
            for d in spec_defs:
                d["Engine"] = "Spectral (FFT)"
                d["Group"] = "Fabric Structure"
                # Sub-classify
                if self._cfg.get("SUB_CLASSIFY", True):
                    d["Type"] = self._sub_classify_spectral(d)
                d["Inspector"] = "Spectral"
            
            defects.extend(spec_defs)
        except Exception as e:
            logger.warning(f"Spectral engine failed: {e}")

        # Engine B: Texture (LBP + Gabor) → additional pattern anomalies
        try:
            img_buffer.seek(0)
            buf_copy = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            
            _, _, ent_map, _, tex_defs = self._texture.detect_defects(
                buf_copy, sensitivity=sensitivity
            )
            viz_maps["entropy"] = ent_map
            
            for d in tex_defs:
                d["Engine"] = "Texture (LBP+Gabor)"
                d["Group"] = "Fabric Structure"
                if self._cfg.get("SUB_CLASSIFY", True):
                    d["Type"] = self._sub_classify_spectral(d)
                d["Inspector"] = "Texture"
            
            defects.extend(tex_defs)
        except Exception as e:
            logger.warning(f"Texture engine failed: {e}")

        # Engine C: Edge/Structural (Canny + Morphology) → Hole, Tear, Snag
        try:
            img_buffer.seek(0)
            buf_copy = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            
            _, _, anomaly_hm, _, edge_defs = self._edge.detect_defects(
                buf_copy, sensitivity=sensitivity
            )
            viz_maps["anomaly_heatmap"] = anomaly_hm
            
            for d in edge_defs:
                d["Engine"] = "Edge (Canny+Morph)"
                d["Group"] = "Fabric Structure"
                if self._cfg.get("SUB_CLASSIFY", True):
                    d["Type"] = self._sub_classify_edge(d)
                d["Inspector"] = "Edge"
            
            defects.extend(edge_defs)
        except Exception as e:
            logger.warning(f"Edge engine failed: {e}")
        
        return defects, viz_maps

    def _route_group_ii(self, img_buffer: BinaryIO, sensitivity: float
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
        """
        Group II: Stitch Quality defects.
        Runs Seam Inspector (which internally runs Projection + Regression + Laplacian).
        """
        defects = []
        viz_maps = {}
        
        try:
            img_buffer.seek(0)
            buf_copy = BytesIO(img_buffer.read())
            img_buffer.seek(0)
            
            _, _, _, seam_output, seam_defs = self._seam.detect_defects(buf_copy)
            viz_maps["seam_output"] = seam_output
            
            for d in seam_defs:
                d["Engine"] = f"Seam ({d.get('Type', 'Unknown')})"
                d["Group"] = "Stitch Quality"
                d["Inspector"] = "Seam"
                # Type is already properly classified by the refactored seam inspector
            
            defects.extend(seam_defs)
        except Exception as e:
            logger.warning(f"Seam engine failed: {e}")
        
        return defects, viz_maps

    # ──────────────────────────────────────────────
    # MAIN API
    # ──────────────────────────────────────────────
    def process(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.5,
        mode: str = "full",
        remove_shadows: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for the Unified Processor.
        
        Args:
            img_buffer: Image file buffer
            sensitivity: Detection sensitivity (1.0 - 5.0)
            mode: "full" | "structure_only" | "seam_only"
            remove_shadows: Whether to apply aggressive background illumination correction
        
        Returns:
            {
                "defects": [...],         # List of all detected defects
                "group_i_defects": [...], # Fabric Structure defects only
                "group_ii_defects": [...],# Stitch Quality defects only
                "viz_maps": {...},        # Visualization maps for UI
                "routing_info": {...},    # Pre-classifier results + engines used
                "summary": {...}          # Counts and verdict
            }
        """
        all_defects = []
        all_viz_maps = {}
        engines_used = []
        
        # Pre-process shadows if requested
        img_buffer.seek(0)
        raw_bytes = img_buffer.read()
        if remove_shadows:
            raw_bytes = self._remove_shadows(raw_bytes)
            
        # Create a new buffer with the processed (or clean) bytes to feed the pipeline
        pipeline_buffer = BytesIO(raw_bytes)
        
        # Pre-classify (determine which groups to run)
        pre_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        img_pre = cv2.imdecode(pre_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img_pre is None:
            raise ValueError("Could not decode image file")
        
        region_info = self._pre_classify(img_pre)
        
        # Determine which groups to run
        run_group_i = mode in ("full", "structure_only") or region_info["has_fabric_body"]
        run_group_ii = mode in ("full", "seam_only") or region_info["has_seam"]
        
        # In "full" mode, always run both regardless of pre-classifier
        if mode == "full":
            run_group_i = True
            run_group_ii = True
        
        # Route to engines using the pipeline_buffer (which is shadow-free if toggled)
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
        
        # Re-number IDs
        for i, d in enumerate(all_defects, 1):
            d["ID"] = i
        
        # Split by group
        group_i = [d for d in all_defects if d.get("Group") == "Fabric Structure"]
        group_ii = [d for d in all_defects if d.get("Group") == "Stitch Quality"]
        
        # Build summary
        total = len(all_defects)
        summary = {
            "total_defects": total,
            "group_i_count": len(group_i),
            "group_ii_count": len(group_ii),
            "verdict": "PASS" if total == 0 else "FAIL",
            "defect_types_found": list(set(d.get("Type", "Unknown") for d in all_defects)),
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
        """Return the full defect taxonomy for UI display."""
        return self.DEFECT_TAXONOMY


# Module instance
unified_processor = UnifiedProcessor()
