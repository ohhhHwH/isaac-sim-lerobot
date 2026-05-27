# 导入前
import argparse
import os

from collections.abc import Sequence

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
import os
import math

# isaacsim 依赖库
import omni
import omni.ui as ui
import isaacsim.core.api
from isaacsim.core.api import World
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, UsdGeom
# isaaclab sim - 仿真相关的工具和配置类
import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
# isaaclab actuators - 执行器相关的配置类和基类
from isaaclab.actuators import ImplicitActuatorCfg
# isaaclab assets - 资产相关的配置类和基类
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.assets import SurfaceGripper, SurfaceGripperCfg
# isaaclab controllers - 控制器相关的配置类和基类
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# isaaclab managers - 场景管理器相关的配置类和基类
from isaaclab.managers import SceneEntityCfg
# isaaclab scene - 场景相关的配置类和基类
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# isaaclab utils - 工具函数和配置类
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms, compute_pose_error
# isaaclab sensors - 传感器相关的配置类和基类
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import CameraCfg, Camera
# isaaclab markers - 可视化标记相关的配置类和基类
from isaaclab.markers import VisualizationMarkersCfg

import h5py
import random

DEBUG_MODE = False  # 调试模式
# 启用后方块位置固定
# 启用后输出日志

USD_PATH = "/home/hyl/isaac-sim-lerobot/assets/piper_isaac_sim/USD/piper_v2_robot.usd"
OBJ_PATH = "/home/hyl/isaac-sim-lerobot/assets/scenes/kitchen_with_orange/assets/Orange001/Orange001.usd"
OBJ2_PATH = "/home/hyl/isaac-sim-lerobot/assets/fruit/green_apple.usd"

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
BODY_NAMES = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]

# camera_link 是什么？
# Available strings: ['arm_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'camera_link', 'link7', 'link8']


ANGLE_STEP = 0.05
PIPER_NAME = "piper"

# 相机参数
CAM_HEIGHT = 480
CAM_WIDTH = 640

# 物体放置参数
OBJ_L = 0.02
OBJ_W = 0.08
OBJ_H = 0.03
OBJ_Z = OBJ_H / 2  # half cube size, sitting on ground
OBJ_PX = 1
OBJ_PY = 0.15
OBJ_PZ = OBJ_Z


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



class KeyboardController:
    def __init__(self):
        self._key_states = {}

    def _on_kb(self, event, *args, **kwargs):
        key = event.input if isinstance(event.input, str) else event.input.name
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS,
                          carb.input.KeyboardEventType.KEY_REPEAT):
            self._key_states[key] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._key_states[key] = False
        return True

    def pop(self, key):
        if self._key_states.get(key, False):
            self._key_states[key] = False
            return True
        return False

    def held(self, key):
        return self._key_states.get(key, False)

    def start(self):
        appwin = omni.appwindow.get_default_app_window()
        inp = carb.input.acquire_input_interface()
        kb = appwin.get_keyboard()
        self._sub = inp.subscribe_to_keyboard_events(kb, self._on_kb)

@configclass
class PiperDemoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    piper = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Piper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=20.0,
                velocity_limit_sim=5.0,
                stiffness=200.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                effort_limit_sim=5.0,
                velocity_limit_sim=5.0,
                stiffness=50.0,
                damping=5.0,
            ),
        },
    )

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
            pos=(0.0, -0.1, 0.5),
            rot=look_at_quat((0.0, -0.1, 0.5), (0.0, 0.0, 0.0)),
            convention="opengl",
        ),
    )

    gripper_cam = CameraCfg(
            # 相机在 USD stage 中的 prim 路径（挂在 gripper_static_1 下）
            prim_path="{ENV_REGEX_NS}/Piper/camera_link/gripper_cam",
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

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(OBJ_L, OBJ_W, OBJ_H),  # 立方体尺寸
            activate_contact_sensors=True,  # 启用 sensors
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.001,  # 质量 0.02kg，适当的质量提高稳定性
                # density=500.0,            # 密度 500kg/m³
            ),
            # 碰撞属性配置 - 关键参数用于提高抓取成功率
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,  # 必须远小于物体最薄维度(1cm)的一半
                rest_offset=0.0,  # 零间隙，紧密接触
                torsional_patch_radius=0.04,
                min_torsional_patch_radius=0.01,
            ),
            # 物理材质属性 - 高摩擦低弹性，便于抓取
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,  # 合理范围：橡胶~1.0-2.0
                dynamic_friction=1.0,
                restitution=0,
                friction_combine_mode="average",  # average 更稳定，避免 multiply 导致极端值
                restitution_combine_mode="max",
            ),
            # 视觉材质
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBJ_PX, OBJ_PY, OBJ_PZ),
        ),
    )
    
    orange = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Orange",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJ_PATH,
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.15, 0.03)),
    )

    apple = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJ2_PATH,
            scale=(0.05, 0.05, 0.05),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.15, 0.08)),
    )

# 创建一个 piperSim 包括 robot 和一些工具函数，放在 isaac_piper.py 中
class PiperSim:
    def __init__(self, scene: InteractiveScene):
        self._scene = scene
        self._sim = scene.sim
        self._robot: Articulation = scene[PIPER_NAME]
        # 初始化逆运动学控制器
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=self._sim.device)
        # 创建 SceneEntityCfg / ee 索引
        self._robot_entity_cfg = SceneEntityCfg(PIPER_NAME, joint_names=["joint[1-6]"], body_names=["link6"])
        self._robot_entity_cfg.resolve(self._scene)
        if self._robot.is_fixed_base:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0] - 1
        else:
            self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0]
        
    # 获取当前关节角度
    def get_joint_angles(self):
        return self._robot.data.joint_pos.clone()
    
    def get_pose(self, end_link="joint6"):
        ecfg = self._robot_entity_cfg
        ee_pose_w = self._robot.data.body_pose_w[:, ecfg.body_ids[0]]
        return ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]

    # link6 仿真坐标系相对真机末端有 +90° Z 旋转偏移，需补偿
    _EE_OFFSET_QUAT = torch.tensor([[0.7071068, 0.0, 0.0, -0.7071068]])

    def get_arm_pos(self):
        """获取末端位姿（base frame），返回 (pos[3], quat_wxyz[4])。
        已补偿 link6 与真机末端的 90° Z 偏移。四元数统一为 w>=0。"""
        root_pose_w = self._robot.data.root_pose_w
        ecfg = self._robot_entity_cfg
        ee_pose_w = self._robot.data.body_pose_w[:, ecfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # 补偿 Z 轴 90° 偏移: Q_real = Q_sim * Q_offset_inv
        offset = self._EE_OFFSET_QUAT.to(ee_quat_b.device)
        ee_quat_b = self._quat_mul(ee_quat_b, offset)
        pos = ee_pos_b.squeeze(0).cpu().tolist()
        quat = ee_quat_b.squeeze(0).cpu().tolist()
        if quat[0] < 0:
            quat = [-v for v in quat]
        return pos, quat

    @staticmethod
    def _quat_mul(a, b):
        """批量四元数乘法 [w,x,y,z], shape (N,4)"""
        w1, x1, y1, z1 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
        w2, x2, y2, z2 = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
        return torch.cat([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=1)

    def get_arm_pos_euler(self):
        """获取末端位姿（base frame），返回 (pos[3], euler_xyz_deg[3])。
        euler 为 extrinsic XYZ 度数，与 Piper SDK RX/RY/RZ 一致。"""
        pos, quat = self.get_arm_pos()
        w, x, y, z = quat
        # extrinsic XYZ euler from quaternion (wxyz)
        rx = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sy = 2*(w*y - z*x)
        sy = np.clip(sy, -1.0, 1.0)
        ry = np.arcsin(sy)
        rz = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return pos, [np.degrees(rx), np.degrees(ry), np.degrees(rz)]
    
    GRIPPER_MAX_OPEN_M = 0.070  # 夹爪最大开口 70mm = 0.07m
    GRIPPER_MAX_RAD = 0.04      # 仿真中夹爪关节最大弧度

    def set_arm_angles(
        self,
        angles_rad: Sequence[float | int] | None = None,
        gripper_open_m: float | int | None = None,
    ) -> bool:
        """
        设置关节角度和夹爪开合，与真机 SDK 录制格式一致。

        Args:
            angles_rad: 6个关节角度，单位弧度（与 CSV 录制值一致：SDK millideg / 1000 * 0.0174533）
            gripper_open_m: 夹爪开口距离，单位米（与 CSV 录制值一致：SDK μm / 1e6）
        """
        joint_pos_t = self._robot.data.joint_pos.clone()

        if angles_rad is not None:
            rads = torch.tensor(list(angles_rad), dtype=torch.float32, device=self._sim.device)
            joint_pos_t[0, :len(rads)] = rads

        if gripper_open_m is not None:
            g = (float(gripper_open_m) / self.GRIPPER_MAX_OPEN_M) * self.GRIPPER_MAX_RAD
            g = max(0.0, min(self.GRIPPER_MAX_RAD, g))
            joint_pos_t[0, 6] = g
            joint_pos_t[0, 7] = g

        joint_vel = torch.zeros_like(joint_pos_t)
        self._robot.write_joint_state_to_sim(joint_pos_t, joint_vel)
        self._robot.set_joint_position_target(joint_pos_t)
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim.cfg.dt)
        return True

    # link6 仿真坐标系相对真机末端有 +90° Z 旋转偏移的逆（用于 set_arm_pos 输入转换）
    _EE_OFFSET_QUAT_INV = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])

    # 末端位姿（base frame），与真机 SDK GetArmEndPoseMsgs 一致
    def set_arm_pos(self, target_pos_b, target_quat_b, gripper_open_m=0.070, solve_steps=1):
        """
        IK 求解并驱动到目标末端位姿。

        Args:
            target_pos_b: 末端位置 [x,y,z]，base frame，单位米
            target_quat_b: 末端姿态四元数 [w,x,y,z]，base frame（真机坐标系）
            gripper_open_m: 夹爪开口距离，单位米
            solve_steps: IK 求解步数，默认1
        """
        # 将真机坐标系四元数转为仿真 link6 坐标系: Q_sim = Q_real * Q_offset_inv^-1 = Q_real * Q_offset
        device = self._sim.device
        quat_t = torch.tensor(target_quat_b, dtype=torch.float32, device=device).reshape(1, 4)
        offset_inv = self._EE_OFFSET_QUAT_INV.to(device)
        quat_sim = self._quat_mul(quat_t, offset_inv)
        target_quat_sim = quat_sim.squeeze(0).tolist()

        trajectory = self.isaac_ik_trace(target_pos_b, target_quat_sim, steps=solve_steps)

        for joint_angles in trajectory:
            self.set_arm_angles(angles_rad=joint_angles[:6], gripper_open_m=gripper_open_m)

    def isaac_ik_trace(self, target_pos_b, target_quat_b, steps=1):
        """
        基于 Link6 末端位姿（base frame），用迭代 DifferentialIK 求解关节角度。
        每步都通过仿真更新 Jacobian 以保证位置和姿态同时收敛。

        Args:
            target_pos_b: 目标末端位置 [x,y,z]，base frame，单位米
            target_quat_b: 目标末端四元数 [w,x,y,z]，base frame
            steps: 插值步数

        Returns:
            list[list[float]]: 长度为 steps 的轨迹点，每项为 6 个关节弧度
        """
        steps = max(int(steps), 1)
        device = self._sim.device
        ecfg = self._robot_entity_cfg

        pos_b = torch.tensor(target_pos_b, dtype=torch.float32, device=device).reshape(1, 3)
        quat_b = torch.tensor(target_quat_b, dtype=torch.float32, device=device).reshape(1, 4)

        root_pose_w = self._robot.data.root_pose_w
        ee_pose_w = self._robot.data.body_pose_w[:, ecfg.body_ids[0]]
        ee_pos_b_start, ee_quat_b_start = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        trajectory = []
        for i in range(steps):
            alpha = (i + 1) / steps
            interp_pos_b = ee_pos_b_start + alpha * (pos_b - ee_pos_b_start)
            ik_command = torch.cat([interp_pos_b, quat_b], dim=1)
            self._ik_controller.reset()
            self._ik_controller.set_command(ik_command)

            # 迭代求解当前子目标（位置+姿态收敛）
            for _ in range(200):
                ee_pose_w = self._robot.data.body_pose_w[:, ecfg.body_ids[0]]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                    ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )

                pos_error, rot_error = compute_pose_error(
                    ee_pos_b, ee_quat_b, interp_pos_b, quat_b, rot_error_type="axis_angle"
                )
                pos_err = torch.norm(pos_error).item()
                rot_err = torch.norm(rot_error).item()

                if pos_err < 0.001 and rot_err < 0.01:
                    break

                jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, ecfg.joint_ids]
                joint_pos = self._robot.data.joint_pos[:, ecfg.joint_ids]
                arm_joint_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

                full_target = self._robot.data.joint_pos.clone()
                full_target[:, ecfg.joint_ids] = arm_joint_des
                self._robot.write_joint_state_to_sim(full_target, torch.zeros_like(full_target))
                self._robot.set_joint_position_target(full_target)
                self._scene.write_data_to_sim()
                self._sim.step()
                self._scene.update(self._sim.cfg.dt)

            trajectory.append(self._robot.data.joint_pos[0, ecfg.joint_ids].tolist())

        return trajectory

    # 修改物体位置
    def set_object_pos(self, obj_name, pos, quat=[1, 0, 0, 0]):
        obj = self._scene[obj_name]
        obj.set_world_pose(pos, quat)

# Rx = roll, Ry = pitch, Rz = yaw
def quat_from_euler_xyz(rx, ry, rz):
    """从 extrinsic XYZ Euler 角（弧度）转为四元数 [w, x, y, z]。
    与 Piper SDK GetArmEndPoseMsgs 的 RX/RY/RZ 对应。"""
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z])

def euler_from_quat(q):
    """四元数 [w,x,y,z] 转 extrinsic XYZ euler (rx, ry, rz) 弧度。"""
    w, x, y, z = q
    rx = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sy = 2*(w*y - z*x)
    sy = np.clip(sy, -1.0, 1.0)
    ry = np.arcsin(sy)
    rz = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return rx, ry, rz

def normalize_quat(q):
    return q / np.linalg.norm(q)

def euler_zyx_to_xyz(euler_zyx_deg):
    """将外部 Piper SDK 的 extrinsic ZYX 欧拉角(度) 转为 get_arm_pos_euler() 的 extrinsic XYZ 欧拉角(度)。
    输入顺序 [rz, ry, rx]（scipy as_euler("zyx") 的输出格式）。"""
    rz, ry, rx = np.radians(euler_zyx_deg)
    # extrinsic ZYX: Q = Qx * Qy * Qz (右乘顺序)
    # 等价于 intrinsic XYZ: Q = Qx * Qy * Qz
    # 直接用 quat_from_euler_xyz(rx, ry, rz) 即可，因为 extrinsic XYZ 和 extrinsic ZYX 的区别在于角度赋值
    # 正确做法：构建旋转 R = Rx(rx) * Ry(ry) * Rz(rz) 对应 extrinsic ZYX
    qx = np.array([np.cos(rx/2), np.sin(rx/2), 0, 0])
    qy = np.array([np.cos(ry/2), 0, np.sin(ry/2), 0])
    qz = np.array([np.cos(rz/2), 0, 0, np.sin(rz/2)])
    # extrinsic ZYX: 先Z再Y再X => Q = Qx * Qy * Qz
    q = quat_mul(quat_mul(qx, qy), qz)
    if q[0] < 0:
        q = -q
    erx, ery, erz = euler_from_quat(q)
    return [np.degrees(erx), np.degrees(ery), np.degrees(erz)]

def quat_mul(a, b):
    """四元数乘法 [w,x,y,z]"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def test_ik():

    # 6个关节角度 + 夹爪开合 ； 末端位姿 [x, y, z, Rx, Ry, Rz]
    
    # 关节角度: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.99) 末端位姿: [[0.06, 0.0, 0.21], [0.0, 85.0, -0.0]]
    # 关节角度: ([45.01, 93.744, -57.566, 0.0, 58.702, 74.98], 0.0) 末端位置: [[0.2, 0.2, 0.2], [-150.03, 0.09, -179.91]]
    # 关节角度: ([0.0, 3.243, 0.0, 0.0, 0.0, 0.0], 0.99) 末端位置: [[0.06, 0.0, 0.21], [0.0, 88.24, -0.0]]
    
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0, # 120Hz 更新频率
        device="cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.4, 0.4], [0.0, 0.0, 0.1])

    scene = InteractiveScene(PiperDemoSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    # 创建 PiperSim 实例
    piper = PiperSim(scene)
    
    # 测试 IK , 循环

    # XJY 读取时关节角度除了个 1000 所以 targets_ang 要 *1000 才是 GetArmJointMsgs 的原始值
    """
        joints = self.piper.GetArmJointMsgs()
        angles_deg = [
            joints.joint_state.joint_1 / self.FACTOR,
            ...
        ]
        
        
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return (
            np.array(
                [end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis],
                dtype=np.float32,
            )
            / self.FACTOR # FACTOR= 1000
            / 1000.0
        )
        
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return R.from_euler(
            "xyz",
            np.array(
                [end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis],
                dtype=np.float32,
            )
            / self.FACTOR, # FACTOR= 1000
            degrees=True,
        ).as_euler("zyx", degrees=True)
    
    """
    targets_ang = [
        # [joint1-6 rad, gripper_open_m] — 与 CSV 录制格式一致
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.070],
        [0.7855, 1.6362, -1.0047, 0.0, 1.0245, 1.3087, 0.0],
        [0.0, 0.0566, 0.0, 0.0, 0.0, 0.0, 0.070]
    ]
    targets_end = [ # zyx 单位 m
        [[0.06, 0.0, 0.21], [0.0, 85.0, -0.0]],
        [[0.2, 0.2, 0.2], [-150.03, 0.09, -179.91]],
        [[0.06, 0.0, 0.21], [0.0, 88.24, -0.0]],
    ]
    
    while simulation_app.is_running():
        for i, (ang, end) in enumerate(zip(targets_ang, targets_end)):
            # 1. 移动到 targets_ang
            piper.set_arm_angles(angles_rad=ang[:6], gripper_open_m=ang[6])
            print(f"\n===== Target {i} =====")
            print(f"设定关节角(rad): {ang[:6]}")

            # 2. 获取末端位姿
            pos, quat = piper.get_arm_pos()
            pos_euler, euler = piper.get_arm_pos_euler()
            print(f"get_arm_pos()       -> pos={pos}, quat={quat}")
            print(f"get_arm_pos_euler() -> pos={pos_euler}, euler={euler}")

            # 3. 对比 get_arm_pos_euler() 与 targets_end-> targets_xyz
            expected_pos, expected_euler = end
            converted_euler = euler_zyx_to_xyz(expected_euler)
            print(f"targets_end(ZYX)    -> pos={expected_pos}, euler={expected_euler}")
            print(f"targets_end转XYZ    -> euler={[round(e,2) for e in converted_euler]}")
            print(f"get_arm_pos_euler() -> euler={[round(e,2) for e in euler]}")
            print(f"  pos误差: {[round(a-b,4) for a,b in zip(pos_euler, expected_pos)]}")
            print(f"  euler误差(转换后): {[round(a-b,2) for a,b in zip(euler, converted_euler)]}")

            # 4. 对比 get_arm_pos_euler() + quat_from_euler_xyz() 与 get_arm_pos() 的 quat
            euler_rad = np.radians(euler)
            quat_from_euler = quat_from_euler_xyz(euler_rad[0], euler_rad[1], euler_rad[2])
            quat_from_euler = normalize_quat(quat_from_euler)
            if quat_from_euler[0] < 0:
                quat_from_euler = -quat_from_euler
            # print(f"quat(from get_arm_pos):   {quat}")
            # print(f"quat(from euler转换):     {quat_from_euler.tolist()}")
            # print(f"  quat差异: {[round(a-b,5) for a,b in zip(quat, quat_from_euler.tolist())]}")

            # 5. 对比 get_arm_pos() + euler_from_quat() 与 get_arm_pos_euler() 的 euler
            euler_back = euler_from_quat(quat)
            euler_back_deg = [np.degrees(e) for e in euler_back]
            # print(f"euler(from get_arm_pos_euler): {euler}")
            # print(f"euler(from quat转换):          {euler_back_deg}")
            # print(f"  euler差异: {[round(a-b,4) for a,b in zip(euler, euler_back_deg)]}")

            # 6. 用 get_arm_pos() 的结果做 IK 求解验证
            # print("--- IK 验证 ---")
            piper.set_arm_pos(pos, quat, gripper_open_m=ang[6]) # 这个好像有问题？
            pos_after, quat_after = piper.get_arm_pos()
            # print(f"IK后 pos={pos_after}, quat={quat_after}")
            # print(f"  IK pos误差: {[round(a-b,5) for a,b in zip(pos, pos_after)]}")
            # print(f"  IK quat误差: {[round(a-b,5) for a,b in zip(quat, quat_after)]}")

            # 
            # IK 求逆
            ik_result = piper.isaac_ik_trace(pos, quat, steps=1)
            
            # 将 ik 的末端位姿 传给 set_arm_angles
            # piper.set_arm_angles(angles_rad=ik_result[-1], gripper_open_m=0.070)

            # 到位后保持 2 秒观察
            for _ in range(240):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.cfg.dt)
        break
    
# 物体随机放置范围（机械臂前方区域）
OBJ_X_RANGE = (0.10, 0.25)
OBJ_Y_RANGE = (-0.10, 0.10)
OBJ_Z_FIXED = 0.03

def test_obj():
    sim_cfg = sim_utils.SimulationCfg(dt=1/120, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.4, 0.4], [0.0, 0.0, 0.1])

    scene = InteractiveScene(PiperDemoSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    stage = get_current_stage()
    orange_prim = stage.GetPrimAtPath("/World/envs/env_0/Orange")
    xform = UsdGeom.Xformable(orange_prim)

    for i in range(10):
        x = random.uniform(*OBJ_X_RANGE)
        y = random.uniform(*OBJ_Y_RANGE)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, OBJ_Z_FIXED))
        print(f"[{i+1}/10] Orange -> ({x:.3f}, {y:.3f}, {OBJ_Z_FIXED})")

        # 保持 3 秒 (120Hz * 3s = 360 steps)
        for _ in range(360):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.cfg.dt)

# 实现 remote control 将 小机械臂的末端位姿 通过 udp 传给仿真，验证 IK 求解和坐标系转换的正确性
IP = "0.0.0.0"
PORT = 3456
CONTROL_MODE = "tor" # "pos" or "tor"
SCALE_POS = 2  # 位置缩放，配合真机坐标系调整
# 需要做个一个 坐标映射
def main():
    import socket
    import json

    sim_cfg = sim_utils.SimulationCfg(dt=1/120, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.6, 0.5], [0.0, 0.0, 0.2])

    scene_cfg = PiperDemoSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    piper = PiperSim(scene)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((IP, PORT))
    print(f"[INFO]: Listening on {IP}:{PORT}, mode={CONTROL_MODE}")

    kb = KeyboardController()
    kb.start()

    while simulation_app.is_running():
        data = None
        try:
            while True:
                data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            pass

        if data is None:
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.cfg.dt)
            continue

        try:
            msg = json.loads(data.decode("utf-8"))

            if CONTROL_MODE == "tor":
                pos = msg.get("position") or msg.get("pos")
                quat = msg.get("orientation") or msg.get("quat")
                print(f"Received pos: {pos}, quat: {quat}")
                gripper = msg.get("gripper_open_m", 0.070)
                # 这里的 pos 和 quat 乘缩放比例后  xyz z高度轴不缩放
                pos = [pos[0] * SCALE_POS, pos[1] * SCALE_POS, pos[2]]  # 只缩放 x 和 y
                piper.set_arm_pos(pos, quat, gripper_open_m=gripper)
            else:
                angles = msg["rad_angles"]
                gripper = msg.get("gripper_open_m", 0.070)
                piper.set_arm_angles(angles_rad=angles, gripper_open_m=gripper)

            if kb.pop("R"):
                sim.reset()
                scene.reset()
                piper = PiperSim(scene)
                print("[INFO]: Reset")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # test_ik()
    test_obj()
    # main()
    simulation_app.close()