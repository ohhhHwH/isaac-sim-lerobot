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
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
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

USD_PATH = "/home/hyl/isaac-sim-lerobot/assets/piper_isaac_sim/USD/piper_h_v1.usd"
OBJ_PATH = "/home/hyl/isaac-sim-lerobot/assets/scenes/kitchen_with_orange/assets/Orange001/Orange001.usd"

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
BODY_NAMES = ["base_link", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "Link8"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]

# camera_link 是什么？
# Available strings: ['base_link', 'Link1', 'Link2', 'Link3', 'Link4', 'Link5', 'Link6', 'camera_link', 'Link7', 'Link8']


ANGLE_STEP = 0.05

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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
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
    orange = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Orange",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJ_PATH,
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.15, 0.03)),
    )

# 创建一个 piperSim 包括 robot 和一些工具函数，放在 isaac_piper.py 中
class PiperSim:
    def __init__(self, scene: InteractiveScene):
        self._scene = scene
        self._sim = scene.sim
        self._robot: Articulation = scene["piper"]
        # 初始化逆运动学控制器
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=self._sim.device)
        # 创建 SceneEntityCfg / ee 索引
        self._robot_entity_cfg = SceneEntityCfg("piper", joint_names=["joint[1-6]"], body_names=["Link6"])
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
    
    # 包括关节角度
    def set_arm(self, joint_pos):
        self._robot.set_joint_position_target(joint_pos)
        # 更新仿真
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim.cfg.dt)

    # 末端位姿，不包括关节角度
    def set_arm_pos(self, target_pos, target_rot, gripper_ang = 0.04):
        # pos(3) 单位 m
        # root_pose shape (N,7) = [pos(3), quat(w,x,y,z)(4)]
        pos = self.isaac_ik_trace(target_pos, target_rot, steps=1)
        target = torch.tensor(pos[0], dtype=torch.float32)
        if gripper_ang is not None:
            target[5] = gripper_ang # 保持当前夹爪状态
        self.set_arm(target)

    def isaac_ik_trace(self, pos, quat=None, steps=1):
        """
        基于 Link6 末端位姿，用 DifferentialIK 求解关节角度。

        Args:
            pos: 目标末端位置 (3,)，world frame，单位 m
            quat: 目标末端四元数 (4,) (w,x,y,z)，world frame。None 则保持当前朝向
            steps: 插值步数，>=1

        Returns:
            list[list[float]]: 长度为 steps 的列表，每项是完整关节目标向量 (8,)
        """
        steps = max(int(steps), 1)
        device = self._sim.device
        ecfg = self._robot_entity_cfg

        pos_t = torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)

        ee_pose_w = self._robot.data.body_pose_w[:, ecfg.body_ids[0]]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        if quat is None:
            quat_t = ee_quat_w.clone()
        else:
            quat_t = torch.tensor(quat, dtype=torch.float32, device=device).unsqueeze(0)

        root_pose_w = self._robot.data.root_pose_w
        root_pos_w = root_pose_w[:, 0:3]
        root_quat_w = root_pose_w[:, 3:7]

        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, pos_t, quat_t
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        full_joint_pos = self._robot.data.joint_pos.clone()
        arm_joint_pos = full_joint_pos[:, ecfg.joint_ids]
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, ecfg.joint_ids]

        trajectory = []
        for i in range(steps):
            alpha = (i + 1) / steps
            interp_pos = ee_pos_b + alpha * (target_pos_b - ee_pos_b)
            interp_quat = target_quat_b

            ik_command = torch.cat([interp_pos, interp_quat], dim=1)
            self._ik_controller.reset()
            self._ik_controller.set_command(ik_command)

            arm_joint_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)

            full_target = full_joint_pos.clone()
            full_target[:, ecfg.joint_ids] = arm_joint_des
            trajectory.append(full_target.squeeze(0).tolist())

            arm_joint_pos = arm_joint_des
            ee_pos_b = interp_pos
            ee_quat_b = interp_quat

        return trajectory


def main():
    pass

# Rx = roll, Ry = pitch, Rz = yaw
def quat_from_euler(pitch, roll, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    q0 = cy * cp * cr + sy * sp * sr
    q1 = cy * sp * cr + sy * cp * sr
    q2 = sy * cp * cr - cy * sp * sr
    q3 = cy * cp * sr - sy * sp * cr
    return np.array([q0, q1, q2, q3])

def euler_from_quat(q):
    q0, q1, q2, q3 = q
    roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
    pitch = np.arcsin(2*(q0*q2 - q3*q1))
    yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))
    return pitch, roll, yaw

def normalize_quat(q):
    return q / np.linalg.norm(q)

def test_ik():
    # 单位 mm   角度单位 °
    # [x, y, z, Rx, Ry, Rz]
    # 321 -10 124 177 8 174 - 向前抓
    # 47 -4 172 170 68 173 - Home位
    # 31 -260 127 177 -4 81 - 向右抓
    # -1 53 177 128 72 141 - home 向左
    
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
    targets = [
        [0.321, -0.010, 0.124, 177, 8, 174], # 向前抓
        [0.047, -0.004, 0.172, 170, 68, 173], # Home位
        [0.031, -0.260, 0.127, 177, -4, 81], # 向右抓
        [-0.001, 0.053, 0.177, 128, 72, 141], # home 向左
    ]
    while simulation_app.is_running():
        for target in targets:
            pos = target[:3]
            rot_euler = target[3:]
            rot_quat = quat_from_euler(*np.radians(rot_euler))
            piper.set_arm_pos(pos, rot_quat, gripper_ang=0.04)
            time.sleep(2.0)  # 等待 2 秒观察结果

if __name__ == "__main__":
    test_ik()
    simulation_app.close()