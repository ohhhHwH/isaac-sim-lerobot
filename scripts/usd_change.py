"""
将 input <usd> 文件中的joint 和 link 替换成 input <urdf> 中的joint 和 link 的位置属性（position）
保留 <usd> 中的关节名和物理属性（如质量、惯量等），以保持物理属性的一致性。
输入保证自由度相同，但关节名可能不匹配，保留 <urdf> 中的关节名，替换位置
输出 urdf 转换后的 usd 文件

该程序存在BUG - 无法成功转出
"""

FILE_USD_PATH = "assets/robots/so101_follower.usd"
FILE_URDF_PATH = "assets/urdf/koch.urdf"
OUTPUT_USD_PATH = "assets/urdf/koch.usd"

import sys
import xml.etree.ElementTree as ET
import argparse
import sys
import os
import json
import numpy as np


# 导入前
# 加载 isaaclab 运行环境
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Koch arm keyboard control")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf


def parse_urdf_transforms(urdf_path):
    """解析 URDF 文件，提取关节和链接的变换信息（含 mesh 路径）"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    joints = {}
    links = {}

    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        origin = joint.find('origin')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')

        if origin is not None:
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
        else:
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]

        joints[joint_name] = {
            'parent': parent,
            'child': child,
            'xyz': xyz,
            'rpy': rpy
        }

    for link in root.findall('link'):
        link_name = link.get('name')
        link_info = {'name': link_name}

        for geom_type in ('visual', 'collision'):
            elem = link.find(geom_type)
            if elem is None:
                continue
            mesh_elem = elem.find('geometry/mesh')
            if mesh_elem is not None:
                filename = mesh_elem.get('filename')
                abs_path = os.path.normpath(os.path.join(urdf_dir, filename))
                link_info[f'{geom_type}_mesh'] = abs_path
                scale_str = mesh_elem.get('scale', '1 1 1')
                link_info[f'{geom_type}_scale'] = [float(s) for s in scale_str.split()]
            origin_elem = elem.find('origin')
            if origin_elem is not None:
                link_info[f'{geom_type}_origin_xyz'] = [float(x) for x in origin_elem.get('xyz', '0 0 0').split()]
                link_info[f'{geom_type}_origin_rpy'] = [float(x) for x in origin_elem.get('rpy', '0 0 0').split()]

        links[link_name] = link_info

    return joints, links


def rpy_to_quaternion(roll, pitch, yaw):
    """将 RPY (roll, pitch, yaw) 转换为四元数"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Gf.Quatf(w, x, y, z)


def find_mesh_prims(prim):
    """在 prim 子树中查找所有 Mesh prim"""
    meshes = []
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            meshes.append(child)
    return meshes


def find_reference_prims(prim):
    """在 prim 子树中查找所有带有 asset reference 的 prim"""
    refs = []
    for child in Usd.PrimRange(prim):
        prim_refs = child.GetReferences()
        metadata = child.GetPrimStack()
        for spec in metadata:
            ref_list = spec.referenceList
            for ref in ref_list.prependedItems:
                if ref.assetPath:
                    refs.append((child, ref.assetPath))
            for ref in ref_list.appendedItems:
                if ref.assetPath:
                    refs.append((child, ref.assetPath))
    return refs


def replace_mesh_asset(prim, new_stl_path):
    """替换 prim 上的 mesh asset reference 为新的 STL 文件路径"""
    old_refs = []
    for spec in prim.GetPrimStack():
        ref_list = spec.referenceList
        for ref in ref_list.prependedItems:
            if ref.assetPath:
                old_refs.append(ref)
        for ref in ref_list.appendedItems:
            if ref.assetPath:
                old_refs.append(ref)

    refs = prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(Sdf.Reference(new_stl_path))
    return old_refs


def update_usd_transforms(usd_path, urdf_joints, urdf_links, output_path):
    """更新 USD 文件中的变换和 mesh 引用"""
    stage = Usd.Stage.Open(usd_path)

    if not stage:
        print(f"错误: 无法打开 USD 文件 {usd_path}")
        return False

    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        root_prim = stage.GetPseudoRoot()

    urdf_joint_names = list(urdf_joints.keys())
    usd_joints = []

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            usd_joints.append(prim)

    print(f"\n找到 {len(usd_joints)} 个 USD 关节")
    print(f"找到 {len(urdf_joint_names)} 个 URDF 关节")

    if len(usd_joints) != len(urdf_joint_names):
        print(f"警告: USD 关节数量 ({len(usd_joints)}) 与 URDF 关节数量 ({len(urdf_joint_names)}) 不匹配")

    # 构建 USD link prim 列表（关节的 parent Xform 即为 link）
    # 第一个 link (base_link) 没有关节指向它，需要单独处理
    usd_link_prims = []
    usd_link_prims.append(root_prim)  # base_link 通常是 root

    for i, usd_joint_prim in enumerate(usd_joints):
        if i >= len(urdf_joint_names):
            break

        urdf_joint_name = urdf_joint_names[i]
        urdf_joint_data = urdf_joints[urdf_joint_name]

        parent_prim = usd_joint_prim.GetParent()

        if parent_prim and parent_prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(parent_prim)
            xformable.ClearXformOpOrder()

            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*urdf_joint_data['xyz']))

            if any(urdf_joint_data['rpy']):
                orient_op = xformable.AddOrientOp()
                quat = rpy_to_quaternion(*urdf_joint_data['rpy'])
                orient_op.Set(Gf.Quatf(quat))

            usd_link_prims.append(parent_prim)

            print(f"更新关节 {i}: USD={usd_joint_prim.GetName()} -> URDF={urdf_joint_name}")
            print(f"  位置: {urdf_joint_data['xyz']}")
            print(f"  旋转 (RPY): {urdf_joint_data['rpy']}")

    # 替换 mesh 引用
    urdf_link_names = list(urdf_links.keys())
    print(f"\n--- 替换 Mesh 引用 ---")
    print(f"URDF links: {urdf_link_names}")

    # 收集 USD 中所有带 mesh 的 link prims（按层级顺序）
    usd_link_with_mesh = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable) and not prim.IsA(UsdPhysics.Joint):
            # 检查此 prim 或其子 prim 是否有 mesh 引用
            refs = find_reference_prims(prim)
            meshes = find_mesh_prims(prim)
            if refs or meshes:
                # 只收集直接包含 mesh 的 link 级别 prim
                already_collected = any(str(prim.GetPath()).startswith(str(p.GetPath()) + "/") for p in usd_link_with_mesh)
                if not already_collected:
                    usd_link_with_mesh.append(prim)

    print(f"USD link prims with mesh: {[str(p.GetPath()) for p in usd_link_with_mesh]}")

    # 按顺序映射 URDF link -> USD link prim，替换 mesh
    for i, urdf_link_name in enumerate(urdf_link_names):
        urdf_link = urdf_links[urdf_link_name]
        visual_mesh = urdf_link.get('visual_mesh')
        collision_mesh = urdf_link.get('collision_mesh')

        if not visual_mesh and not collision_mesh:
            continue

        if i >= len(usd_link_with_mesh):
            print(f"  跳过 {urdf_link_name}: 没有对应的 USD link prim")
            continue

        usd_link_prim = usd_link_with_mesh[i]
        print(f"\n  Link {i}: URDF={urdf_link_name} -> USD={usd_link_prim.GetPath()}")

        # 查找此 link 下的所有 mesh 引用
        for child in Usd.PrimRange(usd_link_prim):
            child_path_lower = str(child.GetPath()).lower()

            # 检查是否有 asset reference
            has_ref = False
            for spec in child.GetPrimStack():
                ref_list = spec.referenceList
                for ref in list(ref_list.prependedItems) + list(ref_list.appendedItems):
                    if ref.assetPath:
                        has_ref = True
                        break

            if not has_ref:
                continue

            # 判断是 visual 还是 collision mesh
            is_collision = 'collision' in child_path_lower
            stl_path = collision_mesh if is_collision else visual_mesh

            if stl_path:
                old_refs = replace_mesh_asset(child, stl_path)
                old_paths = [r.assetPath for r in old_refs]
                print(f"    替换 {child.GetPath()}: {old_paths} -> {stl_path}")

    # 保存修改后的 USD 文件
    stage.GetRootLayer().Export(output_path)
    print(f"\n成功保存到: {output_path}")

    return True


def main():

    usd_path = FILE_USD_PATH
    urdf_path = FILE_URDF_PATH
    output_path = OUTPUT_USD_PATH

    print(f"输入 USD: {usd_path}")
    print(f"输入 URDF: {urdf_path}")
    print(f"输出 USD: {output_path}")

    # 解析 URDF
    print("\n解析 URDF 文件...")
    urdf_joints, urdf_links = parse_urdf_transforms(urdf_path)

    print(f"URDF 关节: {list(urdf_joints.keys())}")
    print(f"URDF 链接: {list(urdf_links.keys())}")

    # 更新 USD
    print("\n更新 USD 变换...")
    success = update_usd_transforms(usd_path, urdf_joints, urdf_links, output_path)

    # 关闭 Isaac Sim
    simulation_app.close()

    if success:
        print("\n完成!")
        sys.exit(0)
    else:
        print("\n失败!")
        sys.exit(1)



if __name__ == "__main__":
    main()