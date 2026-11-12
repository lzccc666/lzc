"""Additional artifact capture; does not change frozen planning or semantics."""
from __future__ import annotations

import numpy as np


def install_grid_capture():
    from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
    original = SemanticNavigationPipeline.save_selection_result
    if getattr(original, '_expansion_grid_capture', False):
        return
    def captured(self, result, output_dir, *args, **kwargs):
        original(self, result, output_dir, *args, **kwargs)
        grid = self._occupancy_grid
        if grid is not None:
            from pathlib import Path
            np.savez_compressed(Path(output_dir)/'captured_live_grid.npz', occupancy=grid.occupancy, origin_xy=grid.origin_xy, resolution=grid.resolution)
    captured._expansion_grid_capture = True
    SemanticNavigationPipeline.save_selection_result = captured
