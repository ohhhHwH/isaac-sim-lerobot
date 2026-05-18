# 导入前
import argparse
import os

# 加载 isaaclab 运行环境
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Koch arm keyboard control")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import carb
import math
import os
import time

# isaacsim 依赖库
import omni
import omni.ui as ui
import isaacsim.core.api
from isaacsim.core.api import World
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import SurfaceGripper, SurfaceGripperCfg

import h5py
import random

DEBUG_MODE = True  # 调试模式
# 启用后方块位置固定

# --- 常量配置 ---
URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "urdf", "koch.urdf"
)
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint_gripper"]
ANGLE_STEP = 0.05  # 每次按键旋转弧度 (~2.9°)
NUM_EXPISODES = 5


OBJ_X_RANGE = (-0.05, 0.05)  # left-right (narrow, centered)
OBJ_Y_RANGE = (0.10, 0.15)  # forward from robot base
OBJ_SIZE_RANGE = (0.02, 0.04)  # cube side length

# OBJ_L = 0.02
# OBJ_W = 0.04
# OBJ_H = 0.02

OBJ_L = 0.02
OBJ_W = 0.08
OBJ_H = 0.03

OBJ_Z = OBJ_H / 2  # half cube size, sitting on ground

OBJ_PX = 1
OBJ_PY = 0.15
OBJ_PZ = OBJ_Z

OBJ_P2X = 0
OBJ_P2Y = 0.15
OBJ_P2Z = OBJ_Z

# Grasp approach parameters
GRIPPER_OFFSET = 0.05  # 夹爪略微偏一点，静态爪与物体不碰撞
PRE_GRASP_HEIGHT_OFFSET = 0.1
GRIPPER_HEIGHT = OBJ_H * 1.3  # 夹爪略微高一点，增加稳定性
LIFT_HEIGHT = 0.1

# Gripper joint values

GRIPPER_OPEN = -1.0
GRIPPER_GRASP = -0.2
GRIPPER_CLOSED = 0.0

# Trajectory interpolation
STEPS_PER_PHASE = 60

# Camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480
# CAM_POS = (0.0, 0.2, 0.0)
# CAM_ROT = (
#     0.766,
#     -0.643,
#     0,
#     0,
# )  # 四元数 (w, x, y, z)，测试为正好看到夹爪 (x:-80,y:0,z:0)


# Place target
PLACE_POS = (0.05, 0.15, 0.05)


# 相机视角的四元数，从哪看到哪
def look_at_quat(
    eye: tuple[float, float, float], target: tuple[float, float, float], reverse=False
):
    """Compute a world-frame camera quaternion (w, x, y, z) looking at target."""
    ex, ey, ez = eye
    tx, ty, tz = target
    fx, fy, fz = tx - ex, ty - ey, tz - ez
    flen = math.sqrt(fx * fx + fy * fy + fz * fz)
    if flen < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    fx, fy, fz = fx / flen, fy / flen, fz / flen

    up_world = (0.0, 0.0, 1.0) if not reverse else (0.0, 0.0, -1.0)
    rx = fy * up_world[2] - fz * up_world[1]
    ry = fz * up_world[0] - fx * up_world[2]
    rz = fx * up_world[1] - fy * up_world[0]
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rlen < 1e-8:
        up_world = (0.0, 1.0, 0.0)
        rx = fy * up_world[2] - fz * up_world[1]
        ry = fz * up_world[0] - fx * up_world[2]
        rz = fx * up_world[1] - fy * up_world[0]
        rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rlen, ry / rlen, rz / rlen

    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx

    m00, m01, m02 = rx, ux, -fx
    m10, m11, m12 = ry, uy, -fy
    m20, m21, m22 = rz, uz, -fz

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


# Koch arm FK 链 (translation_xyz, rotation_axis, axis_sign)
_JOINT_CHAIN = [
    ((0.0, 0.0, 0.039), "z", 1),  # joint1: base yaw
    ((-0.0002, 0.0, 0.0173), "x", -1),  # joint2: shoulder pitch
    ((0.00025, 0.014791, 0.108347), "x", 1),  # joint3: elbow pitch
    ((0.000125, 0.090467, 0.002747), "x", 1),  # joint4: wrist pitch
    ((0.001353, 0.000007, -0.045), "z", -1),  # joint5: wrist roll
    ((-0.0074, -0.00025, -0.01315), "y", -1),  # joint_gripper
]


class KeyboardController:
    """键盘控制器，使用监听按键状态"""

    def __init__(self):
        self._key_states = {}
        self._appwindow = None
        self._input_iface = None
        self._keyboard = None
        self._sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        """carb 键盘事件回调"""
        key_name = event.input if isinstance(event.input, str) else event.input.name
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self._key_states[key_name] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._key_states[key_name] = False
        return True

    def on_press(self, key):
        """检查按键是否刚被按下（按下后返回 True 并清除状态，只触发一次）"""
        if self._key_states.get(key, False):
            self._key_states[key] = False
            return True
        return False

    def on_release(self, key):
        """检查按键是否已释放"""
        return not self._key_states.get(key, False)

    def get(self, key):
        """获取按键当前是否处于按下状态（持续返回 True）"""
        return self._key_states.get(key, False)

    def start(self):
        """开始监听键盘事件"""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input_iface = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input_iface.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def stop(self):
        """停止监听键盘事件"""
        if (
            self._sub is not None
            and self._input_iface is not None
            and self._keyboard is not None
        ):
            self._input_iface.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


class SimIsaacModel:
    """Isaac机械臂仿真器，封装了仿真循环、控制、视角和稳定模式"""

    # 初始化
    def __init__(self, urdf_path):
        self._urdf_path = urdf_path
        self._speed = 1.0
        self._views = {}  # name -> CameraCfg
        self._stable_mode = False
        self._random_objects = []  # Track random objects for cleanup
        self._object_counter = 0  # Counter for unique object names

        # 更新 URDF 路径
        self.KOCH_CFG = ArticulationCfg(
            spawn=sim_utils.UrdfFileCfg(
                asset_path=URDF_PATH,
                activate_contact_sensors=True,  # 启用 sensors
                fix_base=True,  # 固定底座
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(  # 默认使用力控制，配合 PD 增益实现位置控制
                    drive_type="force",  # 使用力控制，配合 PD 增益实现位置控制
                    target_type="position",  # 目标为位置
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(  # 默认 PD 增益，后续可通过 toggle_stable_mode 调整
                        stiffness=100.0,  # 刚度，较高值可减少震荡但可能导致数值不稳定，过高会导致仿真崩溃
                        damping=1.0,  # 阻尼，较高值可减少震荡但可能导致响应变慢
                    ),
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=16,  # 提高：夹爪接触求解更精确
                    solver_velocity_iteration_count=8,
                    fix_root_link=True,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,  # 与物体一致，防止弹出
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),  # 初始位置
            actuators={
                "joint1": ImplicitActuatorCfg(  # Rotation — 基座关节，需要最高刚度
                    joint_names_expr=["joint1"],
                    effort_limit_sim=20.0, # 力矩
                    velocity_limit_sim=5.0, # 速度上限
                    stiffness=400.0, # 刚度系数
                    damping=40.0, # 阻尼系数
                ),
                "joint2": ImplicitActuatorCfg(  # Pitch — 承重关节，需抗重力
                    joint_names_expr=["joint2"],
                    effort_limit_sim=20.0,
                    velocity_limit_sim=5.0,
                    stiffness=400.0,
                    damping=40.0,
                ),
                "joint3": ImplicitActuatorCfg(  # Elbow — 中段关节
                    joint_names_expr=["joint3"],
                    effort_limit_sim=15.0,
                    velocity_limit_sim=5.0,
                    stiffness=200.0,
                    damping=20.0,
                ),
                "joint4": ImplicitActuatorCfg(  # Wrist_Pitch
                    joint_names_expr=["joint4"],
                    effort_limit_sim=10.0,
                    velocity_limit_sim=5.0,
                    stiffness=100.0,
                    damping=10.0,
                ),
                "joint5": ImplicitActuatorCfg(  # Wrist_Roll
                    joint_names_expr=["joint5"],
                    effort_limit_sim=10.0,
                    velocity_limit_sim=5.0,
                    stiffness=100.0,
                    damping=10.0,
                ),
                "gripper": ImplicitActuatorCfg(  # Jaw — 夹爪需要柔顺但足够到位
                    joint_names_expr=["joint_gripper"],
                    effort_limit_sim=5.0,
                    stiffness=50.0,
                    damping=5.0,
                ),
            },
            soft_joint_pos_limit_factor=1.0,  # 软限制因子，允许关节在物理极限附近有一定的超出空间，避免过早触发硬限制导致不稳定
        )

        self.OBJ_CFG = AssetBaseCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(
                    os.path.dirname(__file__),
                    "assets/scenes/kitchen_with_orange/assets/Orange001/Orange001.usd",
                ),
                scale=(0.5, 0.5, 0.5),
            ),
            # 配置质量
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(OBJ_P2X, OBJ_P2Y, OBJ_P2Z)
            ),  # 初始位置
        )

        # 抓取物体属性配置
        # 刚体属性配置
        self._obj_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            linear_damping=2.0,
            angular_damping=2.0,
            max_linear_velocity=1.0,
            max_angular_velocity=57.3,
            disable_gravity=False,
            kinematic_enabled=False,
            max_depenetration_velocity=1.0,  # 降低：防止穿透后弹飞
            solver_position_iteration_count=16,  # 提高：更精确的接触求解
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        )
        # 质量属性配置
        self._obj_mass_props = sim_utils.MassPropertiesCfg(
            mass=0.02,  # 质量 0.02kg，适当的质量提高稳定性
            # density=500.0,            # 密度 500kg/m³
        )
        # 碰撞属性配置 - 关键参数用于提高抓取成功率
        self._obj_collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,  # 必须远小于物体最薄维度(1cm)的一半
            rest_offset=0.0,  # 零间隙，紧密接触
            torsional_patch_radius=0.04,
            min_torsional_patch_radius=0.01,
        )
        # 物理材质属性 - 高摩擦低弹性，便于抓取
        self._obj_physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5,  # 合理范围：橡胶~1.0-2.0
            dynamic_friction=1.0,
            restitution=0,
            friction_combine_mode="average",  # average 更稳定，避免 multiply 导致极端值
            restitution_combine_mode="max",
        )
        # 视觉材质
        self._obj_visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.1, 0.1)
        )

        # 构建场景配置
        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
            )
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0, color=(0.75, 0.75, 0.75)
                ),
            )
            koch = self.KOCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Koch")

            # orange = self.OBJ_CFG.replace(prim_path="{ENV_REGEX_NS}/Orange")

            # Target object (dynamic cuboid, will be randomized)
            # 添加一个物体 摩檫力增大 变成柔体
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(OBJ_L, OBJ_W, OBJ_H),  # 立方体尺寸
                    activate_contact_sensors=True,  # 启用 sensors
                    rigid_props=self._obj_rigid_props,
                    mass_props=self._obj_mass_props,
                    # 碰撞属性配置 - 关键参数用于提高抓取成功率
                    collision_props=self._obj_collision_props,
                    # 物理材质属性 - 高摩擦低弹性，便于抓取
                    physics_material=self._obj_physics_material,
                    # 视觉材质
                    visual_material=self._obj_visual_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(OBJ_PX, OBJ_PY, OBJ_PZ),
                ),
            )

            # 添加 assets/scenes/kitchen_with_orange/assets/Orange001/Orange001.usd
            # orange = RigidObjectCfg(
            #     prim_path="{ENV_REGEX_NS}/Orange",
            #     spawn=sim_utils.UsdFileCfg(
            #         usd_path=os.path.join(os.path.dirname(__file__), "assets/scenes/kitchen_with_orange/assets/Orange001/Orange001.usd"),
            #         activate_contact_sensors=True,
            #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #             rigid_body_enabled=True,
            #             max_linear_velocity=1000.0,
            #             max_angular_velocity=1000.0,
            #             max_depenetration_velocity=100.0,
            #             enable_gyroscopic_forces=True,
            #         ),
            #         mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            #         collision_props=sim_utils.CollisionPropertiesCfg(),
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.1, 0.15, 0.03),  # 放在cube旁边
            #     ),
            # )

            # 挂载在 gripper_static_1 上的相机传感器，严格参考 MuJoCo:
            # <camera name="gripper_cam" pos="0 0.08 0" xyaxes="1 0 0 0 0.8 -0.6"/>
            # 其中 xyaxes 对应旋转矩阵列向量:
            # x=(1,0,0), y=(0,0.8,-0.6), z=x×y=(0,0.6,0.8)
            # 等价为绕 x 轴旋转约 -36.87°，四元数 [w,x,y,z] ≈ [0.948683, -0.316228, 0, 0]
            top = CameraCfg(
                prim_path="{ENV_REGEX_NS}/CameraTop",
                update_period=0.1,
                height=CAM_HEIGHT,
                width=CAM_WIDTH,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, -0.1, 1.0),
                    rot=look_at_quat((0.0, -0.1, 1.0), (0.0, 0.0, 0.0)),
                    convention="opengl",
                ),
            )

            side = CameraCfg(
                prim_path="{ENV_REGEX_NS}/CameraSide",
                update_period=0.1,
                height=CAM_HEIGHT,
                width=CAM_WIDTH,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(1.15, 0.0, 0.2),
                    rot=look_at_quat((1.15, 0.0, 0.2), (0.0, 0.0, 0.12)),
                    convention="opengl",
                ),
            )

            front = CameraCfg(
                prim_path="{ENV_REGEX_NS}/CameraFront",
                update_period=0.1,
                height=CAM_HEIGHT,
                width=CAM_WIDTH,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 1.35, 0.25),
                    rot=look_at_quat((0.0, 1.35, 0.25), (0.0, 0.0, 0.12)),
                    convention="opengl",
                ),
            )

            gripper_cam = CameraCfg(
                # 相机在 USD stage 中的 prim 路径（挂在 gripper_static_1 下）
                prim_path="{ENV_REGEX_NS}/Koch/gripper_static_1/gripper_cam",
                update_period=0.1,  # 传感器输出周期（秒）：每 0.1s 输出一次数据
                height=CAM_HEIGHT,  # 输出图像分辨率（像素）
                width=CAM_WIDTH,
                data_types=["rgb"],  # 需要的输出数据类型（此处只要 RGB 图像）
                # Pinhole 相机模型参数（内参/成像模型）
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,  # 焦距（与 horizontal_aperture 配合计算内参），示例值 24.0
                    focus_distance=400.0,  # 对焦距离（与场景单位一致）
                    horizontal_aperture=20.955,  # 传感器水平孔径（单位与 focal_length 保持一致）
                    clipping_range=(0.01, 1.0e5),  # 裁剪近平面和远平面（场景距离单位）
                ),
                # 相机相对父体的位姿偏移
                offset=CameraCfg.OffsetCfg(
                    pos=(
                        0.0,
                        0.12,
                        0.0,
                    ),  # 平移偏移（x, y, z），单位为场景距离（通常米）
                    # 旋转偏移：四元数 (w, x, y, z)
                    # 注意：四元数方向和符号需要与场景其他部分一致（此处来源于 MuJoCo->Isaac 的映射）
                    rot=look_at_quat(
                        (0.0, 0.12, 0.0), (0.0, 0.0, -0.05), reverse=True
                    ),  # 测试为正好看到夹爪 (x:-67,y:0,z:0)
                    convention="opengl",  # 偏移的解释约定，例如 "world" 表示以世界/绝对参照解释，
                ),
            )

            # --- 夹爪上（两个连杆）被 Cube 施加的力：分别在活动抓手和静态爪体上放传感器 ---
            # 说明：prim_path 指定传感器挂载的 prim（必须唯一对应该环境内的一个 body），
            # filter_prim_paths_expr 用于只报告与哪些 prim 的接触（这里只跟 Cube 的接触会被上报）
            gripper_move_contact_cfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Koch/gripper_moving_1",  # 传感器挂在活动爪体（moving）
                filter_prim_paths_expr=[
                    "{ENV_REGEX_NS}/Cube"
                ],  # 只跟 Cube 的接触被记录
                track_pose=True,  # 是否记录传感器原点位姿（world frame）
                track_contact_points=True,  # 是否记录每个接触点的位置（用于可视化/定位力箭头）
                track_friction_forces=True,  # 是否记录摩擦（切向）分力
                track_air_time=True,  # 是否追踪“空中/接触”时间（需要 force_threshold）
                force_threshold=0.5,  # 小于此合力范数被认为“无接触”（用于 track_air_time）
                debug_vis=True,  # 在场景中画力箭头/接触点，便于调试验证
                update_period=0.0,  # 0.0 表示每个仿真步都更新
                history_length=6,  # 保存的历史帧数（用于平滑/历史查询）
                max_contact_data_count_per_prim=32,  # 每个 prim 最多保存多少个接触记录（避免数据溢出）
                visualizer_cfg=VisualizationMarkersCfg(
                    prim_path="/Visuals/ContactSensorGripperMove",
                    markers={
                        "contact": sim_utils.SphereCfg(
                            radius=0.012,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.4, 0.0)
                            ),
                        ),
                        "no_contact": sim_utils.SphereCfg(
                            radius=0.012,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.8, 0.3)
                            ),
                            visible=False,
                        ),
                    },
                ),
            )

            gripper_static_contact_cfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Koch/gripper_static_1",  # 传感器挂在静态爪体（static）
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                track_air_time=True,
                force_threshold=0.5,
                debug_vis=True,
                update_period=0.0,
                history_length=6,
                max_contact_data_count_per_prim=32,
                visualizer_cfg=VisualizationMarkersCfg(
                    prim_path="/Visuals/ContactSensorGripperStatic",
                    markers={
                        "contact": sim_utils.SphereCfg(
                            radius=0.010,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.6, 1.0)
                            ),
                        ),
                        "no_contact": sim_utils.SphereCfg(
                            radius=0.010,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.5, 0.8, 1.0)
                            ),
                            visible=False,
                        ),
                    },
                ),
            )

            # --- 在 Cube 上放一个传感器，用来观测 Cube 受到夹爪施加的力（便于从被施力对象角度分析） ---
            # 说明：把传感器放在 Cube 上可以直接读取 Cube 受到的合力、接触点和摩擦力（同一 contact 以不同侧上报）
            cube_contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Cube",  # 将传感器放在被抓取物体上（必须精确对应场景中的 Cube prim）
                filter_prim_paths_expr=[
                    "{ENV_REGEX_NS}/Koch/gripper_moving_1",
                    "{ENV_REGEX_NS}/Koch/gripper_static_1",
                ],  # 只关注来自夹爪两个 body 的接触
                track_pose=False,  # 通常物体本身位姿通过 object.data.root_pos_w 可得，传感器不必重复记录
                track_contact_points=True,  # 记录所有接触点位置（用于定位受力位置）
                track_friction_forces=True,  # 记录摩擦力分量
                track_air_time=False,  # 对物体通常不需要追踪 air/contact 时间，可按需打开
                force_threshold=0.5,
                debug_vis=True,  # 在世界中绘制受力箭头（来自 contact_pos_w 和力向量）
                update_period=0.0,
                history_length=6,
                max_contact_data_count_per_prim=32,
                visualizer_cfg=VisualizationMarkersCfg(
                    prim_path="/Visuals/ContactSensorCube",
                    markers={
                        "contact": sim_utils.SphereCfg(
                            radius=0.008,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.7, 0.2, 1.0)
                            ),
                        ),
                        "no_contact": sim_utils.SphereCfg(
                            radius=0.008,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.85, 0.6, 1.0)
                            ),
                            visible=False,
                        ),
                    },
                ),
            )

        self._scene_cfg_class = _SceneCfg

        # 初始化仿真
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 400.0,  # 200Hz：1cm薄物体需要更高频率防穿透
            device=(
                args_cli.device
                if hasattr(args_cli, "device") and args_cli.device
                else "cuda:0"
            ),
            physx=sim_utils.PhysxCfg(
                enable_ccd=True,
                enable_stabilization=True,
                bounce_threshold_velocity=0.2,
            ),
        )

        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim.set_camera_view([0.5, 0.5, 0.5], [0.0, 0.0, 0.15])
        scene_cfg = self._scene_cfg_class(num_envs=1, env_spacing=2.0)
        self._scene = InteractiveScene(scene_cfg)
        self._sim.reset()

        # 添加夹爪表面材质
        gripper_mat_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
            friction_combine_mode="average",
        )
        # 2) 使用正则路径在每个 env 下生成材质 prim（spawn 函数带 @clone）
        material_pattern = f"{self._scene.env_regex_ns}/Koch/physicsMaterial_gripper"
        gripper_mat_cfg.func(material_pattern, gripper_mat_cfg)
        # 3) 找到每个具体的 gripper_static / gripper_moving prim，然后绑定对应的材质 prim
        static_paths = sim_utils.find_matching_prim_paths(
            f"{self._scene.env_regex_ns}/Koch/gripper_static_1"
        )
        moving_paths = sim_utils.find_matching_prim_paths(
            f"{self._scene.env_regex_ns}/Koch/gripper_moving_1"
        )
        # 绑定：对每个具体 prim，材质 prim 路径与其父路径对应
        for static_path in static_paths:
            parent = static_path.rsplit("/", 1)[0]  # e.g. /World/envs/env_0/Koch
            mat_path = f"{parent}/physicsMaterial_gripper"
            sim_utils.bind_physics_material(static_path, mat_path)
        for moving_path in moving_paths:
            parent = moving_path.rsplit("/", 1)[0]
            mat_path = f"{parent}/physicsMaterial_gripper"
            sim_utils.bind_physics_material(moving_path, mat_path)

        # gripper_move = SurfaceGripperCfg(
        #     prim_path="{ENV_REGEX_NS}/Koch/gripper_moving_1",  # 夹爪在场景中的路径表达式
        #     max_grip_distance=0.1,  # 最大夹持距离（米），夹爪与物体表面最大允许距离
        #     shear_force_limit=500.0,  # 剪切力极限（牛顿），超过此力夹爪会松开
        #     coaxial_force_limit=500.0,  # 轴向力极限（牛顿），超过此力夹爪会松开
        #     retry_interval=0.2,  # 夹爪尝试重新夹持的时间间隔（秒）
        # )
        # gripper = SurfaceGripper(gripper_move)
        # self._scene.surface_grippers["gripper"] = gripper

        # 初始化视口和相机系统
        self._viewport = get_viewport_from_window_name("Viewport")
        self._perspective_path = "/OmniverseKit_Persp"  # 默认主视角
        self._view_names = []  # 有序的视角名称列表
        self._current_view_idx = -1  # -1 表示主视角（perspective）
        self._sensor_view_names = set()  # 直接切到已有传感器相机 prim 的视角名

        self._robot: Articulation = self._scene["koch"]
        self._sim_dt = self._sim.get_physics_dt()
        self._target_pos = self._robot.data.default_joint_pos.clone()

        # 预解析 body 索引
        ee_body_ids, _ = self._robot.find_bodies("gripper_static_1")
        self._ee_body_id = ee_body_ids[0]
        gripper_body_ids, _ = self._robot.find_bodies("gripper_moving_1")
        self._gripper_body_id = gripper_body_ids[0]

        # IK 控制器
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self._ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=1, device=self._sim.device
        )
        self._robot_entity_cfg = SceneEntityCfg(
            "koch", joint_names=["joint[1-5]"], body_names=["gripper_static_1"]
        )
        self._robot_entity_cfg.resolve(self._scene)
        self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0] - 1

        # 添加默认视角（机械臂在原点，高约 0.3m，中心约 0.12m）
        self.add_view(
            "top",
            {
                "eye": [0.0, -0.1, 1.3],
                "target": [0.0, 0.0, 0.1],
                "focal_length": 18.0,
            },
        )
        self.add_view(
            "side",
            {
                "eye": [1.15, 0.0, 0.2],
                "target": [0.0, 0.0, 0.12],
            },
        )
        self.add_view(
            "front",
            {
                "eye": [0.0, 1.35, 0.25],
                "target": [0.0, 0.0, 0.12],
            },
        )
        self.add_sensor_view(
            "gripper_cam", "/World/envs/env_0/Koch/gripper_static_1/gripper_cam"
        )
        # self.add_viewport("top")  # 启动时默认显示 top 视角的 viewport TODO 仍有问题，仍需手动添加

        print("[INFO]: SimIsaacModel setup complete.")

    # 设置移动速度
    def set_speed(self, speed):
        self._speed = speed

    # 输入关节角度列表，单位为弧度
    def set_joint_angles(self, joint_angles):
        if isinstance(joint_angles, list):
            joint_angles = torch.tensor(
                joint_angles, dtype=torch.float32, device=self._sim.device
            )
        self._target_pos[0, : len(joint_angles)] = joint_angles
        self._robot.set_joint_position_target(self._target_pos)
        # 步进仿真
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim_dt)

    # 根据末端位姿设定机械臂姿态
    def set_arm_pose(self, target_pos, target_rot, gripper_ang=None):
        # target_pos = pos(3)
        # target_rot = quat(w,x,y,z)(4)
        # TODO 先不设置夹爪开合
        pos = self.isaac_ik_trace(target_pos, target_rot, steps=1)
        target = torch.tensor(pos[0], dtype=torch.float32, device=self._sim.device)
        target[5] = (
            gripper_ang if gripper_ang is not None else self._target_pos[0, 5]
        )  # 保持当前夹爪状态
        self.set_joint_angles(target)

    # 获取当前关节角度列表，单位为弧度
    def get_joint_angles(self, joint_angles=None):
        return self._robot.data.joint_pos[0].cpu().tolist()

    def add_sensor_view(self, view_name, camera_prim_path, view_params=None):
        """添加一个基于场景中已有 camera prim 的视角（如挂在机器人 link 上的传感器相机）。
        view_params: dict
            - eye: [x, y, z] 相机位置（世界坐标）
            - target: [x, y, z] 相机注视点（世界坐标）
            - focal_length: float 焦距，默认 24.0
        """
        self._views[view_name] = {
            "camera_path": camera_prim_path,
        }
        self._view_names.append(view_name)
        self._sensor_view_names.add(view_name)
        print(f"[View] Added sensor view: '{view_name}' path={camera_prim_path}")

    # 添加视角
    def add_view(self, view_name, view_params):
        """
        添加一个可切换的视角。在 USD stage 中创建 Camera prim，通过 V 键切换。
        view_params: dict
            - eye: [x, y, z] 相机位置（世界坐标）
            - target: [x, y, z] 相机注视点（世界坐标）
            - focal_length: float 焦距，默认 24.0
        """
        eye = view_params.get("eye", [0.5, 0.5, 0.5])
        target = view_params.get("target", [0.0, 0.0, 0.15])
        focal_length = view_params.get("focal_length", 24.0)

        # 在 USD stage 中创建 Camera prim
        camera_path = f"/World/ViewCamera_{view_name}"
        stage = get_current_stage()
        camera_prim = stage.DefinePrim(camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(focal_length)
        # 设置 centerOfInterest 属性（Kit 视口需要）
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))

        # 通过 ViewportCameraState 设置相机位置和注视点
        camera_state = ViewportCameraState(camera_path, self._viewport)
        camera_state.set_position_world(Gf.Vec3d(*eye), True)
        camera_state.set_target_world(Gf.Vec3d(*target), True)

        self._views[view_name] = {
            "camera_path": camera_path,
            "eye": eye,
            "target": target,
        }
        self._view_names.append(view_name)
        print(f"[View] Added: '{view_name}' eye={eye} target={target}")

    # 切换视角
    def switch_view(self, view_name=None):
        """
        切换视口到指定视角，如果 view_name 为 None 则循环切换。
        循环顺序: perspective -> view0 -> view1 -> ... -> perspective
        """
        if not self._view_names:
            print("[View] No viewpoints added.")
            return

        if view_name is not None:
            # 切换到指定视角
            if view_name in self._views:
                idx = self._view_names.index(view_name)
                self._current_view_idx = idx
                cam_path = self._views[view_name]["camera_path"]
                self._viewport.set_active_camera(cam_path)
                print(f"[View] Switched to: '{view_name}'")
            else:
                print(f"[View] '{view_name}' not found.")
        else:
            # 循环切换: -1(perspective) -> 0 -> 1 -> ... -> -1
            self._current_view_idx += 1
            if self._current_view_idx >= len(self._view_names):
                # 回到主视角
                self._current_view_idx = -1
                self._viewport.set_active_camera(self._perspective_path)
                print("[View] Switched to: main perspective")
            else:
                name = self._view_names[self._current_view_idx]
                cam_path = self._views[name]["camera_path"]
                self._viewport.set_active_camera(cam_path)
                print(f"[View] Switched to: '{name}'")

    # ui 中 添加一个 viewport，并设置 指定的 view (根据view name) TODO 仍有问题，仍需手动添加
    def add_viewport(self, view_name):
        """
        在 UI 中创建一个新的 3D viewport 窗口，并设置为指定的视角。
        创建独立的 viewport 窗口，不影响主 viewport。

        Parameters
        ----------
        view_name : str
            要显示的视角名称，必须是已通过 add_view() 或 add_sensor_view() 添加的视角

        Returns
        -------
        viewport or None
            成功时返回新创建的 viewport 对象，失败时返回 None
        """
        import asyncio

        if view_name not in self._views:
            print(
                f"[Viewport] Error: view '{view_name}' not found. Available views: {list(self._views.keys())}"
            )
            return None

        camera_path = self._views[view_name]["camera_path"]
        window_name = f"{view_name}_Viewport"

        try:
            # 使用 viewport API 创建新的 3D viewport 窗口
            viewport_api = omni.kit.viewport.utility.get_viewport_interface()

            # 创建新的 viewport 窗口
            viewport_api.create_viewport_window(window_name)

            # 异步等待窗口创建并设置相机
            asyncio.ensure_future(
                self._setup_viewport_camera(window_name, camera_path, view_name)
            )

            print(
                f"[Viewport] Creating viewport window '{window_name}' for camera '{camera_path}'"
            )
            print(f"[Viewport] Window will be docked next to main Viewport once ready")

            return window_name

        except Exception as e:
            print(f"[Viewport] Failed to create viewport window: {e}")
            print(
                f"[Viewport] Tip: You can use switch_view('{view_name}') to change the main viewport instead"
            )
            return None

    async def _setup_viewport_camera(
        self, window_name: str, camera_path: str, view_name: str
    ):
        """异步设置 viewport 的相机并停靠窗口"""
        # 等待窗口创建完成
        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

            # 尝试获取新创建的 viewport
            new_viewport = get_viewport_from_window_name(window_name)

            if new_viewport:
                # 设置相机路径
                new_viewport.set_active_camera(camera_path)
                print(
                    f"[Viewport] Successfully set camera '{camera_path}' for viewport '{window_name}'"
                )

                # 停靠窗口
                await self._dock_viewport_window(window_name)
                return

        print(
            f"[Viewport] Warning: Could not get viewport handle for '{window_name}' after creation"
        )

    async def _dock_viewport_window(self, window_name: str):
        """异步停靠 viewport 窗口到主 Viewport 旁边"""
        # 等待窗口在 workspace 中可用
        for _ in range(5):
            if ui.Workspace.get_window(window_name):
                break
            await omni.kit.app.get_app().next_update_async()

        # 获取窗口引用
        custom_window = ui.Workspace.get_window(window_name)
        viewport_window = ui.Workspace.get_window("Viewport")

        if custom_window and viewport_window:
            # 停靠到主 Viewport 窗口右侧
            custom_window.dock_in(viewport_window, ui.DockPosition.RIGHT, 0.5)
            print(f"[Viewport] Docked '{window_name}' next to main Viewport")
        elif custom_window:
            print(
                f"[Viewport] Window '{window_name}' created but could not dock (main Viewport not found)"
            )
        else:
            print(f"[Viewport] Could not find window '{window_name}' in workspace")

    def joint_angles_to_poses(
        self,
        joint_angles,
        last_joint_body_name="gripper_static_1",
        gripper_body_name="gripper_moving_1",
    ):
        """
        正运动学：根据关节角度计算末端位姿。

        Parameters
        ----------
        joint_angles : list[float]
            各关节角度（弧度），长度应与 model.nq 一致。
        last_joint_body_name : str
            机械臂末端最后一个关节所在连杆的名称，默认为 "gripper_static_1"。
        gripper_body_name : str
            机械臂末端执行器所在连杆的名称，默认为 "gripper_moving_1"。
        Returns
        -------
        dict
            {
                'last_joint': {'pos': np.ndarray(3), 'quat': np.ndarray(4)},
                'gripper':    {'pos': np.ndarray(3), 'quat': np.ndarray(4)},
            }
            last_joint — gripper_static_1（最后一个机械臂关节 joint5 所在连杆）
            gripper    — gripper_moving_1（实际工作点）
            quat 为 Isaac 格式 [w, x, y, z] 四元数
        """
        # 计算到 joint5 (up_to_joint=5) 得到 gripper_static_1 的位姿
        T_last = _make_fk(joint_angles, up_to_joint=5)
        last_pos = T_last[:3, 3].numpy()
        last_quat = _rotation_matrix_to_quat(T_last[:3, :3].numpy())

        # 计算到 joint_gripper (up_to_joint=6) 得到 gripper_moving_1 的位姿
        T_gripper = _make_fk(joint_angles, up_to_joint=6)
        gripper_pos = T_gripper[:3, 3].numpy()
        gripper_quat = _rotation_matrix_to_quat(T_gripper[:3, :3].numpy())

        return {
            "last_joint": {"pos": last_pos, "quat": last_quat},
            "gripper": {"pos": gripper_pos, "quat": gripper_quat},
        }

    def get_current_poses(self):
        """
        从当前仿真状态直接读取末端位姿

        Returns
        -------
        dict  （格式同 joint_angles_to_poses）
        """
        ee_pos = self._robot.data.body_pos_w[0, self._ee_body_id].cpu().numpy()
        ee_quat = (
            self._robot.data.body_quat_w[0, self._ee_body_id].cpu().numpy()
        )  # [w,x,y,z]

        grip_pos = self._robot.data.body_pos_w[0, self._gripper_body_id].cpu().numpy()
        grip_quat = self._robot.data.body_quat_w[0, self._gripper_body_id].cpu().numpy()

        return {
            "last_joint": {"pos": ee_pos, "quat": ee_quat},
            "gripper": {"pos": grip_pos, "quat": grip_quat},
        }

    # 移动到预设的 home 位姿（所有关节角度为 0）
    def move_to_home(self):
        pass

    # isaac sim ik 逆运动学求解 - 并插值，生成平滑轨迹
    def isaac_ik_trace(self, pos, quat=None, rot_rad=0, steps=10):
        """
        参考并使用 IsaacLab 的 DifferentialIKController，求解目标末端位姿对应的关节目标，
        能够在当前关节角和目标关节角之间做线性插值，生成平滑轨迹。

        Args:
            pos: 目标末端位置，世界坐标系，shape=(3,)
            quat: 目标末端四元数 [w, x, y, z]，世界坐标系。默认保持当前末端朝向。
            rot_rad (float, optional): 末端旋转角度（弧度），用于调整末端朝向. Defaults to 0.
            steps (int, optional): 轨迹分段数. Defaults to 10.

        Returns:
            list[list[float]]: 从当前关节角到目标关节角的插值轨迹，每个元素是一组完整关节角。
        """
        steps = max(int(steps), 1)
        device = self._sim.device

        if quat is None and pos is None:
            # raise ValueError("At least one of pos or quat must be provided for IK target.")
            return None

        if pos is None:
            # 保持当前位置不发生改变
            target_pos_w = self._robot.data.body_pos_w[
                :, self._robot_entity_cfg.body_ids[0]
            ]
        else:
            target_pos_w = torch.as_tensor(
                pos, dtype=torch.float32, device=device
            ).reshape(1, 3)

        ee_pos_w = self._robot.data.body_pos_w[:, self._robot_entity_cfg.body_ids[0]]
        ee_quat_w = self._robot.data.body_quat_w[:, self._robot_entity_cfg.body_ids[0]]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w

        if quat is None:
            target_quat_w = ee_quat_w.clone()
        else:
            target_quat_w = torch.as_tensor(
                quat, dtype=torch.float32, device=device
            ).reshape(1, 4)
            quat_norm = torch.linalg.norm(target_quat_w, dim=1, keepdim=True)
            if torch.any(quat_norm < 1e-8):
                raise ValueError("Target quaternion norm must be non-zero.")
            target_quat_w = target_quat_w / quat_norm

        if rot_rad != 0:
            # 构造绕 Z 轴旋转的四元数
            rot_quat = torch.tensor(
                [
                    math.cos(rot_rad / 2),  # w: 实部
                    0,  # x: 绕X轴分量为0
                    0,  # y: 绕Y轴分量为0
                    math.sin(rot_rad / 2),  # z: 绕Z轴分量
                ],
                dtype=torch.float32,
                device=device,
            ).reshape(1, 4)

            # 将新旋转叠加到原朝向上
            target_quat_w = _quat_multiply(rot_quat, target_quat_w)

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )

        ik_command = torch.cat((target_pos_b, target_quat_b), dim=1)
        self._ik_controller.reset()
        self._ik_controller.set_command(ik_command)

        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._robot_entity_cfg.joint_ids
        ]
        joint_pos = self._robot.data.joint_pos[:, self._robot_entity_cfg.joint_ids]
        joint_pos_des = self._ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )

        current_joint_pos = self._robot.data.joint_pos[0].clone()
        target_joint_pos = current_joint_pos.clone()
        target_joint_pos[self._robot_entity_cfg.joint_ids] = joint_pos_des[0]

        trajectory = []
        for i in range(steps):
            alpha = 1.0 if steps == 1 else i / (steps - 1)
            waypoint = current_joint_pos + alpha * (
                target_joint_pos - current_joint_pos
            )
            trajectory.append(waypoint.detach().cpu().tolist())

        return trajectory

    def contact_sensor_infor(self):
        # print information from the sensors
        print("-------------------------------")
        print(self._scene["gripper_move_contact_cfg"])
        print(
            "Received force matrix of: ",
            self._scene["gripper_move_contact_cfg"].data.force_matrix_w,
        )
        print(
            "Received contact force of: ",
            self._scene["gripper_move_contact_cfg"].data.net_forces_w,
        )
        print("-------------------------------")
        print(self._scene["gripper_static_contact_cfg"])
        print(
            "Received force matrix of: ",
            self._scene["gripper_static_contact_cfg"].data.force_matrix_w,
        )
        print(
            "Received contact force of: ",
            self._scene["gripper_static_contact_cfg"].data.net_forces_w,
        )
        print("-------------------------------")
        print(self._scene["cube_contact_forces"])
        print(
            "Received force matrix of: ",
            self._scene["cube_contact_forces"].data.force_matrix_w,
        )
        print(
            "Received contact force of: ",
            self._scene["cube_contact_forces"].data.net_forces_w,
        )
        print("-------------------------------")

    def step(self):
        """执行一步仿真"""
        self._robot.set_joint_position_target(self._target_pos)
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim_dt)
        # if DEBUG_MODE:
        #     self.contact_sensor_infor()

    def reset(self):
        """重置机械臂到默认位置"""
        root_state = self._robot.data.default_root_state.clone()
        root_state[:, :3] += self._scene.env_origins
        self._robot.write_root_pose_to_sim(root_state[:, :7])
        self._robot.write_root_velocity_to_sim(root_state[:, 7:])
        joint_pos = self._robot.data.default_joint_pos.clone()
        joint_vel = self._robot.data.default_joint_vel.clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._robot.reset()
        self._target_pos = self._robot.data.default_joint_pos.clone()

        # 删除场景中添加的随机物体
        self.remove_random_object()

        # 重置物体位置为初始位置
        if "cube" in self._scene.keys():
            cube = self._scene["cube"]
            initial_pos = (
                torch.tensor(
                    [[0.0, 0.15, OBJ_Z]], dtype=torch.float32, device=cube.device
                )
                + self._scene.env_origins
            )
            initial_quat = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=cube.device
            )
            cube.write_root_pose_to_sim(torch.cat((initial_pos, initial_quat), dim=1))
            cube.write_root_velocity_to_sim(
                torch.zeros((1, 6), dtype=torch.float32, device=cube.device)
            )
            print("[Object] Reset cube to initial position")

    def grasp_open(self):
        """打开夹爪"""
        self._target_pos[0, 5] = GRIPPER_OPEN
        self._robot.set_joint_position_target(self._target_pos)

    def close_gripper(self):
        """关闭夹爪"""
        self._target_pos[0, 5] = GRIPPER_CLOSED
        self._robot.set_joint_position_target(self._target_pos)

    def gripper_state(self) -> float:
        """返回夹爪电机的角度"""
        return self._robot.data.joint_pos[0, 5].cpu().item()

    def is_grasping(self, threshold=0.02) -> bool:
        """根据夹爪当前状态判断是否正在夹持物体，threshold 是夹爪闭合的角度阈值（弧度）"""
        gripper_angle = self.gripper_state()
        # If gripper is more closed than threshold from fully open, consider it grasping
        return abs(gripper_angle - GRIPPER_OPEN) > threshold

    def get_object_position(self) -> np.ndarray:
        """获取场景中物体的当前位置

        Returns:
            np.ndarray: 物体的3D位置 [x, y, z]，如果场景中没有物体则返回None
        """
        if "cube" not in self._scene.keys():
            print("[Object] No cube in scene")
            return None

        cube = self._scene["cube"]
        return cube.data.root_pos_w[0].cpu().numpy()

    def randomize_object(self, position=None) -> dict:
        """在场景中的地面上随机(指定范围内)/指定位置添加一个物体，返回物体的坐标
        使用场景中已有的 cube 对象并随机化其位置
        """
        # Get the cube from the scene
        if "cube" not in self._scene.keys():
            print("[Object] No cube in scene, cannot randomize")
            return None

        cube = self._scene["cube"]

        # Generate random position if not specified
        if position is None:
            x = random.uniform(*OBJ_X_RANGE)
            y = random.uniform(*OBJ_Y_RANGE)
            z = OBJ_Z
            position = (x, y, z)
        else:
            x, y, z = position

        # Random orientation (yaw only)
        # yaw = math.pi / 2
        yaw = random.uniform(0, math.pi / 2)
        qw = math.cos(yaw / 2)
        qx, qy = 0.0, 0.0
        qz = math.sin(yaw / 2)
        if DEBUG_MODE:
            # 固定位置和朝向用于调试
            x, y, z = 0.0, 0.15, OBJ_Z
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            position = (x, y, z)

        # Write to simulation
        pose = torch.tensor(
            [[x, y, z, qw, qx, qy, qz]], dtype=torch.float32, device=cube.device
        )
        pose[:, :3] += self._scene.env_origins
        cube.write_root_pose_to_sim(pose)

        # Clear velocity to prevent residual motion
        zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=cube.device)
        cube.write_root_velocity_to_sim(zero_vel)

        print(f"[Object] Randomized cube at ({x:.3f}, {y:.3f}, {z:.3f})")

        # 根据 rot 算出旋转角度
        euler_x, euler_y, euler_z = euler_from_quaternion(qw, qx, qy, qz)
        print(
            f"[Object] Cube quaternion: (w={qw:.3f}, x={qx:.3f}, y={qy:.3f}, z={qz:.3f})"
        )
        print(
            f"[Object] Cube orientation (Euler angles): ({euler_x}°, {euler_y}°, {euler_z}°)"
        )

        return {
            "name": "cube",
            "position": position,
            "orientation": (qw, qx, qy, qz),
            "euler_angles": (euler_x, euler_y, euler_z),
        }

    def remove_random_object(self):
        """移除之前添加的随机物体"""
        if not self._random_objects:
            print("[Object] No random objects to remove")
            return

        stage = get_current_stage()
        for obj_info in self._random_objects:
            prim_path = obj_info["prim_path"]
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.RemovePrim(prim_path)
                print(f"[Object] Removed: {obj_info['name']}")

        self._random_objects.clear()

    def add_object(self, name, position, size=0.03, color=(0.8, 0.1, 0.1)):
        """添加一个物体到场景中，返回物体的坐标等信息"""
        x, y, z = position

        # Random orientation (yaw only)
        yaw = random.uniform(0, 2 * math.pi)
        qw = math.cos(yaw / 2)
        qx, qy = 0.0, 0.0
        qz = math.sin(yaw / 2)

        prim_path = f"/World/envs/env_0/{name}"

        # Create object configuration
        obj_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.CuboidCfg(
                size=(OBJ_L, OBJ_W, OBJ_H),  # 立方体尺寸
                rigid_props=self._obj_rigid_props,
                mass_props=self._obj_mass_props,
                # 碰撞属性配置 - 关键参数用于提高抓取成功率
                collision_props=self._obj_collision_props,
                # 物理材质属性 - 高摩擦低弹性，便于抓取
                physics_material=self._obj_physics_material,
                # 视觉材质
                visual_material=self._obj_visual_material,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x, y, z),
                rot=(qw, qx, qy, qz),
            ),
        )

        # Spawn the object
        obj_cfg.spawn.func(
            prim_path,
            obj_cfg.spawn,
            translation=(x, y, z),
            orientation=(qw, qx, qy, qz),
        )

        print(f"[Object] Added: {name} at ({x:.3f}, {y:.3f}, {z:.3f})")
        # 根据 rot 算出旋转角度
        euler_x, euler_y, euler_z = euler_from_quaternion(qw, qx, qy, qz)

        return {
            "name": name,
            "prim_path": prim_path,
            "position": position,
            "size": size,
            "orientation": (qw, qx, qy, qz),
            "euler_angles": (euler_x, euler_y, euler_z),
        }

    def del_object(self, name):
        """删除场景中的物体"""
        prim_path = f"/World/envs/env_0/{name}"
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            stage.RemovePrim(prim_path)
            print(f"[Object] Deleted: {name}")
            return True
        else:
            print(f"[Object] Not found: {name}")
            return False

    def get_cube(self):
        """获取场景中的 cube 对象"""
        if "cube" in self._scene.keys():
            return self._scene["cube"]
        return None

    def close(self):
        """清理资源"""
        print("[INFO]: Closing SimIsaacModel and cleaning up resources.")
        self.remove_random_object()


# =============================================================================
# FK 辅助函数
# =============================================================================


# 构造并返回一个 4x4 齐次变换矩阵（torch.float32），表示先平移 (tx,ty,tz) 再绕指定轴旋转 angle（右手规则）
def _make_transform(tx, ty, tz, axis, angle):
    """构造 4x4 齐次变换矩阵: Translation(tx,ty,tz) @ Rotation(axis, angle)"""
    c, s = math.cos(angle), math.sin(angle)
    T = torch.eye(4, dtype=torch.float32)
    T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
    if axis == "x":
        T[1, 1], T[1, 2] = c, -s
        T[2, 1], T[2, 2] = s, c
    elif axis == "y":
        T[0, 0], T[0, 2] = c, s
        T[2, 0], T[2, 2] = -s, c
    else:  # z
        T[0, 0], T[0, 1] = c, -s
        T[1, 0], T[1, 1] = s, c
    return T


# 把各关节的平移+旋转顺序组合成末端位姿（位置 + 旋转矩阵）
def _make_fk(joint_angles, up_to_joint=5):
    """正运动学：链式变换计算末端位姿"""
    T = torch.eye(4, dtype=torch.float32)
    for i in range(min(up_to_joint, 6)):
        (tx, ty, tz), axis, sign = _JOINT_CHAIN[i]
        q = float(joint_angles[i])
        T = T @ _make_transform(tx, ty, tz, axis, sign * q)
    return T


# 将旋转矩阵表示转换为四元数，便于在仿真/消息中使用（Isaac/ROS 通常用四元数）
def _rotation_matrix_to_quat(R):
    """旋转矩阵转 [w, x, y, z] 四元数"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def euler_from_quaternion(qw, qx, qy, qz):
    """
    从四元数计算欧拉角（roll, pitch, yaw）
    返回角度为弧度制

    四元数格式: qw + qx*i + qy*j + qz*k
    """

    # Roll (x轴旋转)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y轴旋转) - 使用asin避免数值超出范围
    sinp = 2.0 * (qw * qy - qz * qx)
    # 限制sinp在[-1, 1]范围内防止数值误差
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (z轴旋转)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _quat_multiply(q1, q2):
    """
    四元数乘法: q = q1 * q2
    输入: q1, q2 形状为 (..., 4)，格式 [w, x, y, z]
    返回: 组合后的四元数
    """
    # 提取分量
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # 计算乘积
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # 合并并归一化（防止数值漂移）
    result = torch.stack([w, x, y, z], dim=-1)
    result = result / torch.norm(result, dim=-1, keepdim=True)

    return result


# =============================================================================
# Demo 控制和数据生产
# =============================================================================


# demo controller: 键盘控制机械臂关节角度，V 键切换视角，R 键重置
def demo_control():
    # 创建仿真器实例，加载 URDF 模型
    sim = SimIsaacModel(URDF_PATH)

    # 设置移动速度
    sim.set_speed(0.5)

    # 添加额外视角（__init__ 中已有 top/front/side，可按需追加）
    # sim.add_view("close_up", {
    #     "eye": [0.15, 0.15, 0.2],
    #     "target": [0.0, 0.0, 0.1],
    #     "focal_length": 35.0,
    # })

    # 切换稳定模式
    # sim.toggle_stable_mode(True)

    # 监听键盘输入
    kb = KeyboardController()
    kb.start()

    # 状态变量
    current_joint_idx = 0
    joint_delta = 0.0

    print("\n" + "=" * 50)
    print("Koch Arm Keyboard Control")
    print("=" * 50)
    print("  Joint mode:")
    print("    UP/DOWN    - Select joint")
    print("    LEFT/RIGHT - Decrease/Increase joint angle")
    print("    R          - Reset all joints to zero")
    print("  View:")
    print(
        "    V          - Cycle viewpoints (main -> top -> side -> front -> gripper_cam -> close_up -> main)"
    )
    print("=" * 50)
    print(
        f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})"
    )

    # 主循环
    while simulation_app.is_running():
        """
        Joint mode:
            UP/DOWN    - Select joint
            LEFT/RIGHT - Decrease/Increase joint angle
            R         - Reset all joints to zero
        """

        # 选择关节
        if kb.on_press("UP"):
            current_joint_idx = (current_joint_idx - 1) % len(JOINT_NAMES)
            print(
                f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})"
            )
        if kb.on_press("DOWN"):
            current_joint_idx = (current_joint_idx + 1) % len(JOINT_NAMES)
            print(
                f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})"
            )

        # 调整关节角度（持续按住生效）
        joint_delta = 0.0
        if kb.get("RIGHT"):
            joint_delta = ANGLE_STEP * sim._speed
        if kb.get("LEFT"):
            joint_delta = -ANGLE_STEP * sim._speed

        # 应用角度变化
        if joint_delta != 0.0:
            sim._target_pos[0, current_joint_idx] += joint_delta

        # 切换视角
        if kb.on_press("V"):
            sim.switch_view()

        # 重置
        if kb.on_press("R"):
            sim.reset()
            print("[INFO]: Reset all joints to default position")

        # 步进仿真
        sim.step()

        # 打印末端位姿（每隔一段时间）
        if joint_delta != 0.0:
            poses = sim.get_current_poses()
            ee = poses["last_joint"]
            angles = sim.get_joint_angles()
            angles_str = ", ".join(
                f"{JOINT_NAMES[i]}: {angles[i]:.4f}" for i in range(len(JOINT_NAMES))
            )
            print(
                f"[EE] pos: ({ee['pos'][0]:.4f}, {ee['pos'][1]:.4f}, {ee['pos'][2]:.4f})  "
                f"[Joints] {angles_str}"
            )

    # 清理
    kb.stop()
    # sim.close()

CONTROL_MODE = "tor"

def demo_remote_control():
    # 连接 upd 传输
    # 192.168.2.2:3456 从这个端口中接收数据
    # 接受该端口发送的 关节角度数据,写入到仿真中的目标关节位置
    import socket
    import json

    IP = "0.0.0.0"
    PORT = 3456

    # 初始化模拟器
    sim = SimIsaacModel(URDF_PATH)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((IP, PORT))
    print(f"Listening on {IP}:{PORT} ...")
    kb = KeyboardController()
    kb.start()
    while simulation_app.is_running():
        # 读取缓冲区中所有包，只使用最新的一个，避免处理堆积的旧数据
        data = None
        try:
            while True:
                data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            pass
        if data is None:
            sim.step()
            continue
        try:
            msg = json.loads(data.decode("utf-8"))

            if CONTROL_MODE == "tor":
                pos = msg.get("position") or msg.get("pos")
                rot = msg.get("orientation") or msg.get("quat")
                gripper_angle = msg["gripper_angle"]
                # "position": pos, "orientation": quat, "gripper_angle": gripper_angle
                pos_t = torch.as_tensor([pos], dtype=torch.float32)
                rot_t = torch.as_tensor([rot], dtype=torch.float32)
                sim.set_arm_pose(pos_t, rot_t, gripper_ang=gripper_angle)
            else:
                # print(f"Received from {addr}: {msg}")
                # {'rad_angles': [0.4141748127291231, 0.5890486225480862, 0.2853204265467293, 0.37429131224409645, -0.05062136600022616, -0.1349721288208946]}
                # 处理msg
                joint_angles = msg["rad_angles"]
                print(f"Joint angles: {joint_angles}")

                # 将关节角度写入仿真
                target = torch.tensor(
                    joint_angles, dtype=torch.float32, device=sim._sim.device
                )
                sim._target_pos[0] = target
                sim.step()

            # TODO 当输入 R 时重置 cube 位置
            if kb.on_press("R"):
                sim.reset()
                print("[INFO]: Reset all joints to default position")

        except Exception as e:
            print(f"Error decoding message: {e}")
    kb.stop()


if __name__ == "__main__":
    # demo_control()
    demo_remote_control()
    # 最后关闭
    print("\n[INFO]: Simulation finished, closing application.")
    simulation_app.close()
    print("[INFO]: Application closed successfully.")
