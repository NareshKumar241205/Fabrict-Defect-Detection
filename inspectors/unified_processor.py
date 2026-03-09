# inspectors/unified_processor.py
import logging
from io import BytesIO
from typing import List, Dict, Any, BinaryIO
from inspectors.classic_edge import ClassicEdgeInspector
from inspectors.logic_spectral import LogicSpectralInspector
from inspectors.logic_seam import LogicSeamInspector

logger = logging.getLogger(__name__)

class UnifiedProcessor:
    """Orchestrates parallel execution of Classical CV and Logic Change pipelines."""

    def __init__(self):
        self.classic_edge = ClassicEdgeInspector()
        self.logic_spectral = LogicSpectralInspector()
        self.logic_seam = LogicSeamInspector()

    def _clone_buffer(self, buffer: BinaryIO) -> BytesIO:
        buffer.seek(0)
        return BytesIO(buffer.read())

    def process(self, img_buffer: BinaryIO, sensitivity: float = 2.5, mode: str = "full", remove_shadows: bool = False) -> Dict[str, Any]:
        all_defects = []
        viz_maps = {}

        # ==========================================
        # PIPELINE 1: CLASSICAL CV (Oil, Hole, Tear)
        # ==========================================
        try:
            classic_buf = self._clone_buffer(img_buffer)
            classic_defs, classic_viz = self.classic_edge.detect_defects(classic_buf, sensitivity=sensitivity)
            all_defects.extend(classic_defs)
            viz_maps["classical_edge_map"] = classic_viz
        except Exception as e:
            logger.warning("Pipeline 1 (Classic Edge) failed: %s", e)

        # ==========================================
        # PIPELINE 2: LOGIC CHANGE (Slub, Skip/Miss/Crooked Stitch)
        # ==========================================
        try:
            spectral_buf = self._clone_buffer(img_buffer)
            logic_spec_defs, logic_spec_viz = self.logic_spectral.detect_defects(spectral_buf, sensitivity=sensitivity)
            all_defects.extend(logic_spec_defs)
            viz_maps["logic_spectral_map"] = logic_spec_viz
        except Exception as e:
            logger.warning("Pipeline 2 (Logic Spectral) failed: %s", e)

        try:
            seam_buf = self._clone_buffer(img_buffer)
            logic_seam_defs, logic_seam_viz = self.logic_seam.detect_defects(seam_buf)
            all_defects.extend(logic_seam_defs)
            viz_maps["logic_seam_map"] = logic_seam_viz
        except Exception as e:
            logger.warning("Pipeline 2 (Logic Seam) failed: %s", e)

        # ==========================================
        # COMBINE AND RETURN
        # ==========================================
        for i, d in enumerate(all_defects, 1):
            d["ID"] = i

        return {
            "defects": all_defects,
            "viz_maps": viz_maps,
            "routing_info": {},  # Placeholder for routing information
            "summary": {
                "total_defects": len(all_defects),
                "defect_types_found": sorted(set(d.get("Type", "Unknown") for d in all_defects))
            }
        }

unified_processor = UnifiedProcessor()
