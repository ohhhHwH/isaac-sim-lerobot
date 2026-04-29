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

def set_arm(robot: Articulation, joint_pos):
    robot.set_joint_position_target(joint_pos)

def set_arm_pos(robot: Articulation, target_pos, target_rot):
    pass

def main():
    pass

def test_usd():
    pass

if __name__ == "__main__":
    main()
    simulation_app.close()