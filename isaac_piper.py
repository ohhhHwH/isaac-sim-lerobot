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
USD_PATH = "assets/urdf/piper_h_v1.usd"
OBJ_PATH = "assets/scenes/kitchen_with_orange/assets/Orange001/Orange001.usd"

def main():
    pass

def test_usd():
    pass

if __name__ == "__main__":
    main()
    simulation_app.close()