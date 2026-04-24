#!/usr/bin/env python3
"""检查 USD 文件结构的工具脚本"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect USD file structure")
parser.add_argument("--usd_path", type=str, default="urdf/ref/so100.usd", help="Path to USD file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics
import os

# 加载 USD 文件
usd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args_cli.usd_path)
print(f"\n{'='*60}")
print(f"Inspecting USD file: {usd_path}")
print(f"{'='*60}\n")

stage = Usd.Stage.Open(usd_path)

if not stage:
    print(f"ERROR: Failed to open USD file: {usd_path}")
    simulation_app.close()
    exit(1)

print("Prim hierarchy:")
print("-" * 60)

def print_prim_info(prim, indent=0):
    """递归打印 prim 信息"""
    prefix = "  " * indent
    prim_type = prim.GetTypeName()

    # 检查是否有 ArticulationRootAPI
    has_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    articulation_marker = " [ARTICULATION ROOT]" if has_articulation else ""

    # 检查是否是关节
    is_joint = prim.IsA(UsdPhysics.Joint) or "Joint" in prim_type
    joint_marker = " [JOINT]" if is_joint else ""

    print(f"{prefix}{prim.GetPath()} ({prim_type}){articulation_marker}{joint_marker}")

    # 如果是关节，打印关节信息
    if is_joint:
        for attr in prim.GetAttributes():
            if "joint" in attr.GetName().lower():
                print(f"{prefix}  - {attr.GetName()}: {attr.Get()}")

# 遍历所有 prim
for prim in stage.Traverse():
    depth = len(str(prim.GetPath()).split('/')) - 2
    print_prim_info(prim, depth)

print("\n" + "="*60)
print("Summary:")
print("-" * 60)

# 查找所有关节
joints = []
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint) or "Joint" in prim.GetTypeName():
        joints.append(str(prim.GetPath()))

print(f"Found {len(joints)} joints:")
for joint in joints:
    print(f"  - {joint}")

# 查找 articulation root
articulation_roots = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_roots.append(str(prim.GetPath()))

print(f"\nFound {len(articulation_roots)} articulation roots:")
for root in articulation_roots:
    print(f"  - {root}")

if not articulation_roots:
    print("\nWARNING: No ArticulationRootAPI found in USD file!")
    print("You may need to:")
    print("  1. Add ArticulationRootAPI to the root prim in the USD file")
    print("  2. Or convert from URDF which automatically adds this API")

print("\n" + "="*60)

simulation_app.close()
