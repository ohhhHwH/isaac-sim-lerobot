"""
修复 green_apple.usd 中的贴图路径问题。
1. 从 GLB 中提取内嵌贴图到 textures/ 目录
2. 修改 USD 中的贴图引用路径指向本地 ./textures/object_0_texture0.png
"""

import struct
import json
import os
from pxr import Usd, UsdShade, Sdf

ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
FRUIT_DIR = os.path.join(ASSET_DIR, "assets", "fruit")
GLB_PATH = os.path.join(FRUIT_DIR, "object_0.glb")
USD_PATH = os.path.join(FRUIT_DIR, "green_apple.usd")
TEXTURE_DIR = os.path.join(FRUIT_DIR, "textures")
TEXTURE_FILENAME = "object_0_texture0.png"

# Step 1: 从 GLB 中提取贴图
os.makedirs(TEXTURE_DIR, exist_ok=True)
texture_out = os.path.join(TEXTURE_DIR, TEXTURE_FILENAME)

with open(GLB_PATH, "rb") as f:
    f.read(4)  # magic
    f.read(4)  # version
    f.read(4)  # length
    # JSON chunk
    chunk_len = struct.unpack("<I", f.read(4))[0]
    f.read(4)  # chunk type
    gltf = json.loads(f.read(chunk_len))
    # Binary chunk
    bin_len = struct.unpack("<I", f.read(4))[0]
    f.read(4)  # chunk type
    bin_data = f.read(bin_len)

image_bv_index = gltf["images"][0]["bufferView"]
bv = gltf["bufferViews"][image_bv_index]
offset = bv.get("byteOffset", 0)
img_data = bin_data[offset : offset + bv["byteLength"]]

with open(texture_out, "wb") as f:
    f.write(img_data)
print(f"[1/2] 提取贴图: {texture_out} ({len(img_data)} bytes)")

# Step 2: 修改 USD 中的贴图路径
stage = Usd.Stage.Open(USD_PATH)
for prim in stage.Traverse():
    if prim.IsA(UsdShade.Shader):
        shader = UsdShade.Shader(prim)
        for inp in shader.GetInputs():
            val = inp.Get()
            if isinstance(val, Sdf.AssetPath) and TEXTURE_FILENAME in val.path:
                new_path = f"./textures/{TEXTURE_FILENAME}"
                inp.Set(Sdf.AssetPath(new_path))
                print(f"[2/2] 修改路径: {val.path} -> {new_path}")

stage.GetRootLayer().Save()
print("完成! USD 已保存。")
