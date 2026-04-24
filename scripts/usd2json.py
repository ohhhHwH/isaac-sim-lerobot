import argparse
import sys
import os
import json
try:
    import yaml
except Exception:
    yaml = None
# 导入前

# 加载 isaaclab 运行环境
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Koch arm keyboard control")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd


# 需要 isaaclab_arena 包目录下运行

from isaaclab_arena.utils.usd_helpers import open_stage, get_all_prims, get_prim_depth, is_articulation_root, is_rigid_body
from isaaclab_arena.utils.usd_pose_helpers import get_prim_pose_in_default_prim_frame
from isaaclab_arena.assets.object_utils import detect_object_type

def serialize_value(v):
    if v is None:
        return None
    if isinstance(v, (str, bool, int, float)):
        return v
    # sequences (but not strings/bytes)
    try:
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            # Normalize to list for easy processing
            seq = list(v)
            # If the numeric sequence is very long, represent it compactly
            if len(seq) > 50 and all(_is_number(x) for x in seq):
                return "..."
            # If sequence is numeric (ints/floats/bools), perform run-length
            # encoding: consecutive repeated values become "count:val".
            def _is_number(x):
                return isinstance(x, (int, float, bool))

            if len(seq) > 0 and all(_is_number(x) for x in seq):
                compressed = []
                prev = seq[0]
                count = 1
                for x in seq[1:]:
                    if x == prev:
                        count += 1
                    else:
                        if count > 1:
                            compressed.append(f"{count}:{serialize_value(prev)}")
                        else:
                            compressed.append(serialize_value(prev))
                        prev = x
                        count = 1
                # finalize last run
                if count > 1:
                    compressed.append(f"{count}:{serialize_value(prev)}")
                else:
                    compressed.append(serialize_value(prev))
                return compressed
            # fallback: recursively serialize each element
            return [serialize_value(x) for x in seq]
    except Exception:
        pass
        # helper: detect numeric-like values, include numpy generic if available
        def _is_number(x):
            try:
                import numpy as _np
            except Exception:
                _np = None
            types = (int, float, bool)
            if _np is not None:
                types = types + (_np.generic,)
            return isinstance(x, types)

        # helper: compress a numeric sequence with run-length encoding
        def _compress_numeric_sequence(sq):
            if len(sq) > 50 and all(_is_number(x) for x in sq):
                return "..."
            compressed = []
            prev = sq[0]
            count = 1
            for x in sq[1:]:
                if x == prev:
                    count += 1
                else:
                    if count > 1:
                        compressed.append(f"{count}:{serialize_value(prev)}")
                    else:
                        compressed.append(serialize_value(prev))
                    prev = x
                    count = 1
            # finalize last run
            if count > 1:
                compressed.append(f"{count}:{serialize_value(prev)}")
            else:
                compressed.append(serialize_value(prev))
            return compressed

        # If top-level sequence is numeric, compress it directly
        if len(seq) > 0 and all(_is_number(x) for x in seq):
            return _compress_numeric_sequence(seq)

        # Otherwise, serialize each element; if an element is itself a sequence
        # and numeric, compress that inner sequence as well.
        out_list = []
        for x in seq:
            # treat nested sequences (but not strings/bytes)
            if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
                try:
                    inner = list(x)
                except Exception:
                    out_list.append(serialize_value(x))
                    continue
                if len(inner) > 0 and all(_is_number(y) for y in inner):
                    out_list.append(_compress_numeric_sequence(inner))
                else:
                    out_list.append([serialize_value(y) for y in inner])
            else:
                out_list.append(serialize_value(x))
        return out_list
    try:
        # fallback: try to convert numpy / torch scalars
        import numpy as _np
        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    try:
        import torch as _t
        if isinstance(v, _t.Tensor):
            return v.detach().cpu().tolist()
    except Exception:
        pass
    return str(v)

def inspect_stage(stage):
    out = {}
    default = stage.GetDefaultPrim()
    out["default_prim"] = str(default.GetPath()) if default else None
    # detect object type (uses stage param as allowed by repo helper)
    try:
        out["detected_object_type"] = detect_object_type(stage=stage).name
    except Exception as e:
        out["detected_object_type_error"] = str(e)
    prims = []
    for prim in get_all_prims(stage):
        p = {}
        p["path"] = str(prim.GetPath())
        p["typeName"] = prim.GetTypeName()
        p["depth"] = get_prim_depth(prim)
        p["applied_schemas"] = list(prim.GetAppliedSchemas())
        p["is_articulation_root"] = bool(is_articulation_root(prim))
        p["is_rigid_body"] = bool(is_rigid_body(prim))
        # attributes
        attrs = {}
        for attr in prim.GetAttributes():
            name = str(attr.GetName())
            try:
                val = attr.Get()
                attrs[name] = serialize_value(val)
            except Exception:
                attrs[name] = "<unreadable>"
        p["attributes"] = attrs
        # pose in default prim frame
        try:
            pose = get_prim_pose_in_default_prim_frame(prim, stage)
            p["pose"] = {
                "position_xyz": list(pose.position_xyz),
                "rotation_wxyz": list(pose.rotation_wxyz),
            }
        except Exception as e:
            p["pose_error"] = str(e)
        prims.append(p)
    out["prims"] = prims
    return out


def extract_physics_properties(stage):
    """返回包含几何、质量、碰撞、刚体相关属性的 prim 列表。

    每个条目包含 prim 路径、类型名、applied_schemas，以及匹配到的 geometry/physics 属性字典。
    """
    results = []
    try:
        from pxr import UsdGeom, UsdPhysics
    except Exception:
        UsdGeom = None
        UsdPhysics = None

    for prim in get_all_prims(stage):
        entry = {
            "path": str(prim.GetPath()),
            "typeName": prim.GetTypeName(),
            "applied_schemas": list(prim.GetAppliedSchemas()),
        }
        geom_props = {}
        phys_props = {}

        for attr in prim.GetAttributes():
            name = str(attr.GetName())
            lname = name.lower()
            try:
                val = serialize_value(attr.Get())
            except Exception:
                val = "<unreadable>"

            # 可能与几何有关的属性名关键词
            if any(k in lname for k in ("point", "extent", "facevertex", "normal", "primvars", "st", "uv", "positions", "indices")):
                geom_props[name] = val
                continue

            # 可能与物理/碰撞/刚体有关的属性名关键词
            if any(k in lname for k in ("mass", "inertia", "density", "collision", "rigid", "physics", "physx", "friction", "restitution", "dynamic", "body", "centerofmass", "com")):
                phys_props[name] = val
                continue

        # 还可根据 applied schemas 判断（例如包含 Physics、Rigid、Collision 等）
        schemas = entry["applied_schemas"]
        if any("physics" in s.lower() or "rigid" in s.lower() or "collision" in s.lower() or "mass" in s.lower() for s in schemas):
            # 若 schemas 指示有物理相关，但没有属性读到，则记录 schemas 以便后续检查
            if not phys_props:
                phys_props["_schemas"] = schemas

        if geom_props or phys_props:
            entry["geometry"] = geom_props
            entry["physics"] = phys_props
            results.append(entry)

    return results


def extract_rigidobject_cfgs(stage):
    """从 stage 中提取类似 RigidObjectCfg 的配置信息。

    返回列表，每项包含:
      - prim_path
      - spawn: { size, activate_contact_sensors, rigid_props, mass_props, collision_props, physics_material, visual_material }
      - init_state: { pos }
    使用启发式规则从 prim 的 attributes / applied_schemas / transform 提取信息。
    """
    cfgs = []
    for prim in get_all_prims(stage):
        try:
            if not (is_rigid_body(prim) or
                    "rigid" in prim.GetTypeName().lower() or
                    any("rigid" in s.lower() or "physics" in s.lower() or "collision" in s.lower() for s in prim.GetAppliedSchemas())):
                continue
        except Exception:
            continue

        cfg = {
            "prim_path": str(prim.GetPath()),
            "spawn": {},
            "init_state": {},
        }

        # collect candidate attributes
        attrs = list(prim.GetAttributes())

        # helper to find attribute by keyword
        def find_attrs_by_kw(keywords):
            res = {}
            for a in attrs:
                name = str(a.GetName())
                lname = name.lower()
                if any(k in lname for k in keywords):
                    try:
                        res[name] = serialize_value(a.Get())
                    except Exception:
                        res[name] = "<unreadable>"
            return res

        # size: look for size/dimensions/extent
        size_candidates = find_attrs_by_kw(("size", "dimension", "extent", "dimensions"))
        if size_candidates:
            # choose first candidate and try normalize to 3-tuple
            first_val = next(iter(size_candidates.values()))
            cfg["spawn"]["size"] = first_val

        # contact sensors
        contact_candidates = find_attrs_by_kw(("contact", "sensor"))
        if contact_candidates:
            # if any attribute indicates activation true, set activate_contact_sensors
            vals = list(contact_candidates.values())
            cfg["spawn"]["activate_contact_sensors"] = any(v is True or v == "True" or v == 1 for v in vals)

        # rigid/mass/collision/physics/visual materials
        cfg["spawn"]["rigid_props"] = find_attrs_by_kw(("rigid", "rigid_props", "rigidprops"))
        cfg["spawn"]["mass_props"] = find_attrs_by_kw(("mass", "inertia", "centerofmass", "com", "density"))
        cfg["spawn"]["collision_props"] = find_attrs_by_kw(("collision", "collider", "colliders", "contact"))
        cfg["spawn"]["physics_material"] = find_attrs_by_kw(("physics_material", "physic", "material", "friction", "restitution"))
        cfg["spawn"]["visual_material"] = find_attrs_by_kw(("visual", "appearance", "shader", "material", "albedo", "basecolor"))

        # transform / initial position
        try:
            pose = get_prim_pose_in_default_prim_frame(prim, stage)
            cfg["init_state"]["pos"] = list(pose.position_xyz)
        except Exception:
            # fallback: look for translate attrs
            trans = find_attrs_by_kw(("xformop:translate", "translate", "position", "pos"))
            if trans:
                cfg["init_state"]["pos"] = next(iter(trans.values()))

        cfgs.append(cfg)

    return cfgs
usd_paths = [
    "urdf/ref/so_100.usd",
    "urdf/ref/so100.usd",
    "urdf/ref/task_cube.usd",
    ]
for usd_path in usd_paths:
    out_path = usd_path + ".inspect.yaml"
    with open_stage(usd_path) as stage:
        data = inspect_stage(stage)
    with open(out_path, "w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(data, f, indent=2, allow_unicode=True, sort_keys=False)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote inspection YAML to: {out_path}")

    # 额外：提取几何/物理属性并保存
    physics_out = usd_path + ".physics.yaml"
    with open_stage(usd_path) as stage:
        phys_data = extract_physics_properties(stage)
    with open(physics_out, "w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(phys_data, f, indent=2, allow_unicode=True, sort_keys=False)
        else:
            json.dump(phys_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote physics YAML to: {physics_out}")

    # 提取 RigidObjectCfg 风格配置并保存
    rigid_out = usd_path + ".rigid_cfgs.yaml"
    with open_stage(usd_path) as stage:
        rigid_data = extract_rigidobject_cfgs(stage)
    with open(rigid_out, "w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(rigid_data, f, indent=2, allow_unicode=True, sort_keys=False)
        else:
            json.dump(rigid_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote rigid cfgs YAML to: {rigid_out}")


# 最后关闭
simulation_app.close()