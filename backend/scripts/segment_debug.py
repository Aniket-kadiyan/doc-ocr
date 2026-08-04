"""
Reproduction harness for auto-segment.

Runs an image through detect_regions + the full clustering chain, prints every
proposed box and every resulting cluster (received-image pixel coords), and
writes an annotated overlay so detection-misses vs clustering-misses are visible
at a glance.

Usage (from backend/, under the venv):
    ./.venv/bin/python scripts/segment_debug.py /path/to/crop.png

Outputs:
    <crop>.detect.png   — raw detect_regions proposals (cyan)
    <crop>.clusters.png — final clusters after merge/split/expand (red, numbered)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr_pipeline import OcrPipeline  # noqa: E402
from region_cluster import (  # noqa: E402
    cluster_boxes,
    merge_fragment_clusters,
    order_clusters,
    split_mixed_clusters,
    union_bbox,
)


def _draw_box(d: ImageDraw.ImageDraw, x, y, w, h, color, label=None) -> None:
    d.rectangle([x, y, x + w, y + h], outline=color, width=2)
    if label is not None:
        d.text((x + 2, max(0, y - 11)), str(label), fill=color)


def main(path: str) -> None:
    img = Image.open(path).convert("RGB")
    print(f"image: {path}  size={img.size}")
    pipe = OcrPipeline()

    boxes = pipe.detect_regions(img)
    print(f"\n=== detect_regions: {len(boxes)} proposals ===")
    for i, b in enumerate(boxes):
        print(
            f"  [{i:2d}] x={b['x']:.0f} y={b['y']:.0f} w={b['w']:.0f} h={b['h']:.0f}"
            f"  aspect={b['w']/max(b['h'],1):.2f}  text={b.get('text','')!r}"
        )

    det = img.copy()
    dd = ImageDraw.Draw(det)
    for i, b in enumerate(boxes):
        _draw_box(dd, b["x"], b["y"], b["w"], b["h"], (0, 200, 220), i)
    out_det = str(Path(path).with_suffix(".detect.png"))
    det.save(out_det)

    iw, ih = img.size
    clustered = cluster_boxes(boxes, margin_ratio=0.72, img_w=iw, img_h=ih)
    clustered = merge_fragment_clusters(clustered)
    clusters = order_clusters(split_mixed_clusters(clustered))
    clusters = pipe._expand_clusters(img, clusters, cluster_margin=0.72)

    print(f"\n=== clusters: {len(clusters)} ===")
    clu = img.copy()
    cd = ImageDraw.Draw(clu)
    for i, c in enumerate(clusters):
        ub = union_bbox(c)
        print(
            f"  ({i+1}) x={ub['x']:.0f} y={ub['y']:.0f} "
            f"w={ub['width']:.0f} h={ub['height']:.0f}  members={len(c)}"
        )
        _draw_box(cd, ub["x"], ub["y"], ub["width"], ub["height"], (220, 30, 30), i + 1)
    out_clu = str(Path(path).with_suffix(".clusters.png"))
    clu.save(out_clu)

    print(f"\noverlays written:\n  {out_det}\n  {out_clu}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: segment_debug.py /path/to/crop.png", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
