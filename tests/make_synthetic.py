"""Fake MoNuSAC-layout data for smoke tests: <root>/<patient>/<patient>_<k>.tif + .xml (incl. touching nuclei + an Ambiguous region)."""
import os
import sys
import numpy as np
import cv2

CLASSES = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]


def make(root, n_patients=6, imgs_per=2, seed=0, test_patients=0):
    rng = np.random.default_rng(seed)
    for p in range(n_patients + test_patients):
        is_test = p >= n_patients
        pid = f"TCGA-AA-{1000 + p}-01Z-00-DX1"
        d = os.path.join(root, "MoNuSAC_test" if is_test else "MoNuSAC_train", pid)
        os.makedirs(d, exist_ok=True)
        tint = rng.uniform(0.8, 1.1, 3)
        for k in range(1, imgs_per + 1):
            h, w = int(rng.integers(200, 420)), int(rng.integers(200, 420))
            img = (np.clip(rng.normal([225, 190, 215], 8, (h, w, 3)) * tint, 0, 255)).astype(np.uint8)
            xml = ['<Annotations>']
            rid = 0
            for a in range(int(rng.integers(8, 25))):
                cx, cy = rng.integers(20, w - 20), rng.integers(20, h - 20)
                r = int(rng.integers(6, 14))
                cname = CLASSES[int(rng.integers(0, 4))] if a % 7 else "Ambiguous"
                for off in ([0, 0], [r + 2, 0]) if a % 3 == 0 else ([0, 0],):  # touching pair
                    c = (cx + off[0], cy + off[1])
                    cv2.circle(img, c, r, (110, 70, 140), -1)
                    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                    verts = "".join(f'<Vertex X="{c[0] + r * np.cos(x):.1f}" Y="{c[1] + r * np.sin(x):.1f}"/>' for x in t)
                    rid += 1
                    xml.append(f'<Annotation Id="{rid}"><Attributes><Attribute Name="{cname}" Id="0"/></Attributes>'
                               f'<Regions><RegionAttributeHeaders/><Region Id="1"><Vertices>{verts}</Vertices></Region></Regions></Annotation>')
            xml.append('</Annotations>')
            stem = f"{pid}_{k}"
            cv2.imwrite(os.path.join(d, stem + ".tif"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            open(os.path.join(d, stem + ".xml"), "w").write("".join(xml))


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_monusac", test_patients=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
