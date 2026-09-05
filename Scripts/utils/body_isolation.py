"""
Body-only vertebral isolation from the TotalSegmentator `vertebrae_body` mask.

Added 2026-09 after the adversarial review (finding F01). The original
`01_segmentation.isolate_vertebral_body` kept the largest connected component
per axial slice of the whole-vertebra label, which in pedicle slices is the
entire vertebra (body + arch). The `vertebrae_body` task gives the bodies
without the posterior arch, but as ONE binary mask for all vertebrae, so the
work here is:

1. split it into instances (connected components, with a marker-watershed
   split for the rare case where two adjacent bodies are bridged);
2. assign instances to the pipeline's already-selected vertebrae ("L1" =
   superior, "L2" = inferior) by overlap with the whole-vertebra labels that
   `vertebra_detection` chose (this keeps the sub-114 Z-split override in
   force: the override decides WHICH vertebrae, the body mask decides the SHAPE);
3. apply the 10 % endplate exclusion to the BODY z-extent (the original used
   the whole-vertebra extent, which includes the spinous process).

Everything is returned with a QC record so a human can check the assignment.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed

MIN_INSTANCE_VOXELS = 2000       # ~0.6 cm3; smaller pieces are fragments
INTERIOR_ERODE_MM = 1.5          # erosion used to find separate interiors of a bridged component
MIN_INTERIOR_VOXELS = 1000
ENDPLATE_EXCLUDE_PERCENT = 10.0


@dataclass
class InstanceInfo:
    instance_id: int
    voxels: int
    volume_cm3: float
    z_min: int
    z_max: int
    z_centroid: float
    touches_z_edge: bool
    overlap_L1: int
    overlap_L2: int
    assigned_to: str = ""            # "L1", "L2" or ""
    from_split: bool = False


@dataclass
class BodyIsolationResult:
    success: bool
    instances: List[InstanceInfo]
    splits: List[dict]
    assignment: Dict[str, int]
    body_z_range: Dict[str, Tuple[int, int]]
    endplate_slices_excluded: Dict[str, int]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["body_z_range"] = {k: list(v) for k, v in self.body_z_range.items()}
        return d


def _bbox(mask: np.ndarray, pad: int = 1):
    idx = np.where(mask)
    sl = []
    for ax in range(3):
        lo = max(int(idx[ax].min()) - pad, 0)
        hi = min(int(idx[ax].max()) + pad + 1, mask.shape[ax])
        sl.append(slice(lo, hi))
    return tuple(sl)


def split_bridged_instances(body: np.ndarray, spacing) -> Tuple[np.ndarray, List[dict]]:
    """
    Label the binary body mask. Components that contain more than one substantial
    interior after a 1.5 mm erosion are split by marker watershed on the distance
    transform, confined to the original component (no voxels are added).
    """
    raw, n = ndi.label(body)
    counts = np.bincount(raw.ravel())
    out = np.zeros(body.shape, dtype=np.int16)
    next_id = 1
    splits = []
    for rid in range(1, n + 1):
        if counts[rid] < MIN_INSTANCE_VOXELS:
            continue
        comp = raw == rid
        sl = _bbox(comp, 1)
        crop = np.pad(comp[sl], 1)
        dt = ndi.distance_transform_edt(crop, sampling=spacing)
        seeds, ns = ndi.label(dt > INTERIOR_ERODE_MM)
        sizes = np.bincount(seeds.ravel())
        ids = [i for i in range(1, ns + 1) if sizes[i] > MIN_INTERIOR_VOXELS]
        if len(ids) > 1:
            markers = np.zeros(seeds.shape, np.int16)
            for j, i in enumerate(ids, 1):
                markers[seeds == i] = j
            parts = watershed(-dt, markers, mask=crop)[1:-1, 1:-1, 1:-1]
            dest = out[sl]
            new_ids = []
            for j in range(1, len(ids) + 1):
                dest[parts == j] = next_id
                new_ids.append(next_id)
                next_id += 1
            splits.append({"raw_component": int(rid), "raw_voxels": int(counts[rid]),
                           "separate_interiors": len(ids), "new_instance_ids": new_ids})
        else:
            out[comp] = next_id
            next_id += 1
    return out, splits


def exclude_endplates(mask: np.ndarray, percent: float = ENDPLATE_EXCLUDE_PERCENT):
    z = np.where(mask.any(axis=(0, 1)))[0]
    if len(z) == 0:
        return mask.copy(), 0
    z_min, z_max = int(z.min()), int(z.max())
    n = int((z_max - z_min + 1) * percent / 100.0)
    out = mask.copy()
    if z_min + n >= z_max - n:      # degenerate: keep at least the central slice
        n = max((z_max - z_min) // 2 - 1, 0)
    out[:, :, :z_min + n] = False
    out[:, :, z_max - n + 1:] = False
    return out, n


def isolate_bodies(body_mask: np.ndarray,
                   label_masks: Dict[str, np.ndarray],
                   spacing,
                   endplate_percent: float = ENDPLATE_EXCLUDE_PERCENT
                   ) -> Tuple[Dict[str, np.ndarray], BodyIsolationResult]:
    """
    Parameters
    ----------
    body_mask   : binary vertebrae_body mask (all bodies in the FOV)
    label_masks : {"L1": whole-vertebra mask selected as superior target,
                   "L2": whole-vertebra mask selected as inferior target}
                  (from vertebra_detection, incl. any subject override)
    spacing     : voxel size (mm)

    Returns
    -------
    bodies : {"L1": body-only mask after endplate exclusion, "L2": ...}
    result : BodyIsolationResult with the full QC record
    """
    warnings = []
    labels, splits = split_bridged_instances(body_mask.astype(bool), spacing)
    counts = np.bincount(labels.ravel())
    vox_vol = float(np.prod(spacing)) / 1000.0
    nz = body_mask.shape[2]
    split_ids = {i for s in splits for i in s["new_instance_ids"]}

    instances: List[InstanceInfo] = []
    for i in range(1, labels.max() + 1):
        if counts[i] == 0:
            continue
        m = labels == i
        z = np.where(m.any(axis=(0, 1)))[0]
        instances.append(InstanceInfo(
            instance_id=i, voxels=int(counts[i]), volume_cm3=round(counts[i] * vox_vol, 2),
            z_min=int(z.min()), z_max=int(z.max()), z_centroid=float(np.mean(np.where(m)[2])),
            touches_z_edge=bool(z.min() == 0 or z.max() == nz - 1),
            overlap_L1=int(np.count_nonzero(m & label_masks["L1"])),
            overlap_L2=int(np.count_nonzero(m & label_masks["L2"])),
            from_split=i in split_ids))

    if len(instances) < 2:
        return {}, BodyIsolationResult(False, instances, splits, {}, {}, {},
                                       [f"only {len(instances)} body instances found"])

    # One-to-one assignment maximising overlap with the selected whole-vertebra labels
    ids = [inst.instance_id for inst in instances]
    cost = -np.array([[inst.overlap_L1 for inst in instances],
                      [inst.overlap_L2 for inst in instances]], dtype=float)
    rows, cols = linear_sum_assignment(cost)
    assignment = {}
    for r, c in zip(rows, cols):
        lvl = ["L1", "L2"][r]
        inst = instances[c]
        if -cost[r, c] == 0:
            warnings.append(f"{lvl}: no body instance overlaps the selected label")
            continue
        inst.assigned_to = lvl
        assignment[lvl] = inst.instance_id
        frac = (-cost[r, c]) / inst.voxels
        if frac < 0.5:
            warnings.append(f"{lvl}: only {100*frac:.0f}% of body instance {inst.instance_id} "
                            f"lies inside the selected label (label may be incomplete)")
        if inst.touches_z_edge:
            warnings.append(f"{lvl}: body instance {inst.instance_id} touches the z-edge of the FOV")
        # any other instance with substantial overlap with the same label = fragment/split label
        others = [o for o in instances if o.instance_id != inst.instance_id
                  and getattr(o, f"overlap_{lvl}") > 0.2 * o.voxels]
        for o in others:
            warnings.append(f"{lvl}: additional body instance {o.instance_id} ({o.volume_cm3} cm3) also "
                            f"overlaps this label; not merged")

    if len(assignment) < 2:
        return {}, BodyIsolationResult(False, instances, splits, assignment, {}, {}, warnings)

    # Sanity: superior must be above inferior
    zc = {lvl: next(i.z_centroid for i in instances if i.instance_id == iid) for lvl, iid in assignment.items()}
    if zc["L1"] <= zc["L2"]:
        warnings.append("assigned L1 body is not superior to L2 body")

    bodies, zr, excl = {}, {}, {}
    for lvl, iid in assignment.items():
        full = labels == iid
        z = np.where(full.any(axis=(0, 1)))[0]
        zr[lvl] = (int(z.min()), int(z.max()))
        bodies[lvl], excl[lvl] = exclude_endplates(full, endplate_percent)

    return bodies, BodyIsolationResult(True, instances, splits, assignment, zr, excl, warnings)
