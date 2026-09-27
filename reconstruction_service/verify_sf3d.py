"""Independent SF3D check (no backend involved).
Usage (SF3D venv): python verify_sf3d.py <image> [output.glb]"""
import sys
import time

from sf3d_runner import Sf3dRunner

runner = Sf3dRunner()
t = time.perf_counter()
runner.load()
print("load:", round(time.perf_counter() - t, 1), "s", runner.info())
image = open(sys.argv[1], "rb").read()
result = runner.reconstruct(image)
out = sys.argv[2] if len(sys.argv) > 2 else "sf3d_verify.glb"
open(out, "wb").write(result["glb"])
print("timings:", result["timings"], "peak MB:", result["peakMemoryMb"], "glb bytes:", len(result["glb"]), "->", out)
