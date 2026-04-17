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
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sensors import ContactSensorCfg


import h5py
import random

DEBUG_MODE = True  # 调试模式
# 启用后方块位置固定

# --- 常量配置 ---
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "koch.urdf")
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint_gripper"]
ANGLE_STEP = 0.05  # 每次按键旋转弧度 (~2.9°)
NUM_EXPISODES = 2

# Object randomization ranges (from data_produce.py)
OBJ_X_RANGE = (-0.05, 0.05)   # left-right (narrow, centered)
OBJ_Y_RANGE = (0.10, 0.15)    # forward from robot base
OBJ_Z = 0.015                 # half cube size, sitting on ground
OBJ_SIZE_RANGE = (0.02, 0.04)  # cube side length

OBJ_L = 0.005
OBJ_W = 0.08
OBJ_H = 0.02
GRIPPER_OFFSET = 0.02 # 夹爪略微偏一点，静态爪与物体不碰撞

# Gripper joint values

GRIPPER_OPEN = -1.0
GRIPPER_GRASP = 0
GRIPPER_CLOSED = 0.0

# Trajectory interpolation
STEPS_PER_PHASE = 60

# Camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_POS = (0.0, 0.2, 0.0)
CAM_ROT = (0.766, -0.643, 0, 0) # 四元数 (w, x, y, z)，测试为正好看到夹爪 (x:-80,y:0,z:0)


# Place target
PLACE_POS = (0.05, 0.15, 0.05)

# Grasp approach parameters
PRE_GRASP_HEIGHT_OFFSET = 0.1
PRE_GRASP_HEIGHT_OFFSET = 0.06
LIFT_HEIGHT = 0.15

# Koch arm FK 链 (translation_xyz, rotation_axis, axis_sign)
_JOINT_CHAIN = [
    ((0.0, 0.0, 0.039),              'z',  1),   # joint1: base yaw
    ((-0.0002, 0.0, 0.0173),         'x', -1),   # joint2: shoulder pitch
    ((0.00025, 0.014791, 0.108347),   'x',  1),   # joint3: elbow pitch
    ((0.000125, 0.090467, 0.002747),  'x',  1),   # joint4: wrist pitch
    ((0.001353, 0.000007, -0.045),    'z', -1),   # joint5: wrist roll
    ((-0.0074, -0.00025, -0.01315),   'y', -1),   # joint_gripper
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
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
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
        if self._sub is not None and self._input_iface is not None and self._keyboard is not None:
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
                activate_contact_sensors=True, # 启用 sensors
                fix_base=True,# 固定底座
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg( # 默认使用力控制，配合 PD 增益实现位置控制
                    drive_type="force", # 使用力控制，配合 PD 增益实现位置控制
                    target_type="position", # 目标为位置
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg( # 默认 PD 增益，后续可通过 toggle_stable_mode 调整
                        stiffness=400.0, # 刚度，较高值可减少震荡但可能导致数值不稳定，过高会导致仿真崩溃
                        damping=40.0, # 阻尼，较高值可减少震荡但可能导致响应变慢
                    ),
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg( # 机械臂根部属性
                    enabled_self_collisions=True, # 启用自碰撞，防止机械臂自身穿透
                    solver_position_iteration_count=8, # 增加迭代次数提高稳定性，减少震荡
                    solver_velocity_iteration_count=0, # 速度迭代通常不需要，保持默认
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg( # 刚体属性
                    disable_gravity=False, # 启用重力
                    max_depenetration_velocity=5.0, # 限制最大分离速度，防止穿透后弹出过快导致不稳定
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)), # 初始位置
            actuators={
                "all_joints": ImplicitActuatorCfg( # 统一配置所有关节的 actuator，方便后续调整 PD 增益实现稳定模式切换
                    joint_names_expr=["joint[1-5]"], # 匹配所有关节
                    effort_limit_sim=5.0, # 仿真中的力矩限制，过高可能导致不稳定，过低可能无法驱动机械臂
                    velocity_limit_sim=5.0, # 仿真中的速度限制，过高可能导致不稳定，过低可能导致响应变慢
                    stiffness=400.0, # 默认刚度，较高值可减少震荡但可能导致数值不稳定，过高会导致仿真崩溃
                    damping=60.0, # 默认阻尼，较高值可减少震荡但可能导致响应变慢
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["joint_gripper"], # gripper joint
                    effort_limit_sim=10,   # 限制更小的力矩
                    # 关键：低刚度 + 高阻尼 = 柔顺控制，允许夹爪在接触时有一定的顺应性，减少对物体的冲击和震荡，提高抓取成功率
                    stiffness=200.0,          # 0 → 夹爪不被位置伺服强制到目标（变得顺从/合规）
                    damping=120,            # 小阻尼避免完全失控震荡
                ),
            },
        )

        # 抓取物体属性配置
        # 刚体属性配置
        self._obj_rigid_props = sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=1.0,      # 线性阻尼：减少平移运动的晃动
                        angular_damping=1.0,     # 角阻尼：减少旋转运动的晃动
                        max_linear_velocity=1.0, # 最大线速度限制 (m/s)
                        max_angular_velocity=57.3, # 最大角速度限制 (deg/s) ≈ 1 rad/s
                        disable_gravity=False,   # 启用重力
                        kinematic_enabled=False,      # 是否为运动学物体
                        
                        # 高级稳定性参数
                        max_depenetration_velocity=10.0,  # 最大分离速度，防止穿透后弹出
                        solver_position_iteration_count=4,  # 位置求解迭代次数
                        solver_velocity_iteration_count=1,  # 速度求解迭代次数
                        sleep_threshold=0.005,        # 休眠阈值，低于此速度进入休眠节省计算
                        stabilization_threshold=0.001, # 稳定化阈值
                    )
        # 质量属性配置
        self._obj_mass_props=sim_utils.MassPropertiesCfg(
            mass=0.02,               # 质量 0.02kg，适当的质量提高稳定性
            # density=500.0,            # 密度 500kg/m³
        )
        # 碰撞属性配置 - 关键参数用于提高抓取成功率
        self._obj_collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,    # 接触偏移 (m)：碰撞检测开始的距离
            rest_offset=0.001,         # 静止偏移 (m)：物体静止时的间隙，0表示紧密接触
            torsional_patch_radius=0.04,  # 扭转摩擦接触半径 (m)
            min_torsional_patch_radius=0.01, # 最小扭转摩擦半径 (m)
            
            # 高级碰撞参数
            # collision_enabled=True,      # 是否启用碰撞（默认True）
        )
        # 物理材质属性 - 高摩擦低弹性，便于抓取
        self._obj_physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=100,     # 静摩擦系数：物体静止时的摩擦力，越大越不易滑动
            dynamic_friction=80,    # 动摩擦系数：物体运动时的摩擦力
            restitution=0,         # 弹性系数：0表示完全非弹性碰撞（不反弹）
            friction_combine_mode="multiply",  # 摩擦力组合模式：multiply表示相乘
            restitution_combine_mode="min",    # 弹性组合模式：min表示取最小值
            # 高级摩擦参数
            # friction_restitute=0.0,      # 摩擦恢复
            # improve_patch_friction=True,  # 改进接触面摩擦计算
        )
        # 视觉材质
        self._obj_visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1))

        # 构建场景配置
        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
            )
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            )
            koch = self.KOCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Koch")

            # Target object (dynamic cuboid, will be randomized) 
            # 添加一个物体 摩檫力增大 变成柔体
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(OBJ_L, OBJ_W, OBJ_H),  # 立方体尺寸
                    activate_contact_sensors=True, # 启用 sensors
                    rigid_props=self._obj_rigid_props,
                    mass_props=self._obj_mass_props,
                    # 碰撞属性配置 - 关键参数用于提高抓取成功率
                    collision_props=self._obj_collision_props,
                    # 物理材质属性 - 高摩擦低弹性，便于抓取
                    physics_material=self._obj_physics_material,
                    # 视觉材质
                    visual_material=self._obj_visual_material
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.15, OBJ_Z),
                ),
            )

            # 挂载在 gripper_static_1 上的相机传感器，严格参考 MuJoCo:
            # <camera name="gripper_cam" pos="0 0.08 0" xyaxes="1 0 0 0 0.8 -0.6"/>
            # 其中 xyaxes 对应旋转矩阵列向量:
            # x=(1,0,0), y=(0,0.8,-0.6), z=x×y=(0,0.6,0.8)
            # 等价为绕 x 轴旋转约 -36.87°，四元数 [w,x,y,z] ≈ [0.948683, -0.316228, 0, 0]
            gripper_cam = CameraCfg(
                # 相机在 USD stage 中的 prim 路径（挂在 gripper_static_1 下）
                prim_path="{ENV_REGEX_NS}/Koch/gripper_static_1/gripper_cam",
                update_period=0.1, # 传感器输出周期（秒）：每 0.1s 输出一次数据
                height=CAM_HEIGHT, # 输出图像分辨率（像素）
                width=CAM_WIDTH,
                data_types=["rgb"], # 需要的输出数据类型（此处只要 RGB 图像）
                # Pinhole 相机模型参数（内参/成像模型）
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, # 焦距（与 horizontal_aperture 配合计算内参），示例值 24.0
                    focus_distance=400.0, # 对焦距离（与场景单位一致）
                    horizontal_aperture=20.955, # 传感器水平孔径（单位与 focal_length 保持一致）
                    clipping_range=(0.01, 1.0e5), # 裁剪近平面和远平面（场景距离单位）
                ),
                # 相机相对父体的位姿偏移
                offset=CameraCfg.OffsetCfg(
                    pos=CAM_POS, # 平移偏移（x, y, z），单位为场景距离（通常米）
                    # 旋转偏移：四元数 (w, x, y, z)
                    # 注意：四元数方向和符号需要与场景其他部分一致（此处来源于 MuJoCo->Isaac 的映射）
                    rot=CAM_ROT, # 测试为正好看到夹爪 (x:-80,y:0,z:0)
                    convention="opengl", # 偏移的解释约定，例如 "world" 表示以世界/绝对参照解释，
                ),
            )

            
            # --- 夹爪上（两个连杆）被 Cube 施加的力：分别在活动抓手和静态爪体上放传感器 ---
            # 说明：prim_path 指定传感器挂载的 prim（必须唯一对应该环境内的一个 body），
            # filter_prim_paths_expr 用于只报告与哪些 prim 的接触（这里只跟 Cube 的接触会被上报）
            gripper_move_contact_cfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Koch/gripper_moving_1",   # 传感器挂在活动爪体（moving）
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],     # 只跟 Cube 的接触被记录
                track_pose=True,               # 是否记录传感器原点位姿（world frame）
                track_contact_points=True,     # 是否记录每个接触点的位置（用于可视化/定位力箭头）
                track_friction_forces=True,    # 是否记录摩擦（切向）分力
                track_air_time=True,           # 是否追踪“空中/接触”时间（需要 force_threshold）
                force_threshold=0.5,           # 小于此合力范数被认为“无接触”（用于 track_air_time）
                debug_vis=True,                # 在场景中画力箭头/接触点，便于调试验证
                update_period=0.0,             # 0.0 表示每个仿真步都更新
                history_length=6,              # 保存的历史帧数（用于平滑/历史查询）
                max_contact_data_count_per_prim=32,  # 每个 prim 最多保存多少个接触记录（避免数据溢出）
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
            )

            # --- 在 Cube 上放一个传感器，用来观测 Cube 受到夹爪施加的力（便于从被施力对象角度分析） ---
            # 说明：把传感器放在 Cube 上可以直接读取 Cube 受到的合力、接触点和摩擦力（同一 contact 以不同侧上报）
            cube_contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Cube",  # 将传感器放在被抓取物体上（必须精确对应场景中的 Cube prim）
                filter_prim_paths_expr=[
                    "{ENV_REGEX_NS}/Koch/gripper_moving_1",
                    "{ENV_REGEX_NS}/Koch/gripper_static_1",
                ],  # 只关注来自夹爪两个 body 的接触
                track_pose=False,              # 通常物体本身位姿通过 object.data.root_pos_w 可得，传感器不必重复记录
                track_contact_points=True,     # 记录所有接触点位置（用于定位受力位置）
                track_friction_forces=True,    # 记录摩擦力分量
                track_air_time=False,          # 对物体通常不需要追踪 air/contact 时间，可按需打开
                force_threshold=0.5,
                debug_vis=True,                # 在世界中绘制受力箭头（来自 contact_pos_w 和力向量）
                update_period=0.0,
                history_length=6,
                max_contact_data_count_per_prim=32,
            )


        self._scene_cfg_class = _SceneCfg

        # 初始化仿真
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device if hasattr(args_cli, 'device') and args_cli.device else "cuda:0")
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim.set_camera_view([0.5, 0.5, 0.5], [0.0, 0.0, 0.15])
        scene_cfg = self._scene_cfg_class(num_envs=1, env_spacing=2.0)
        self._scene = InteractiveScene(scene_cfg)
        self._sim.reset()

        


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
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=self._sim.device)
        self._robot_entity_cfg = SceneEntityCfg("koch", joint_names=["joint[1-5]"], body_names=["gripper_static_1"])
        self._robot_entity_cfg.resolve(self._scene)
        self._ee_jacobi_idx = self._robot_entity_cfg.body_ids[0] - 1
        
        
        # 添加默认视角（机械臂在原点，高约 0.3m，中心约 0.12m）
        self.add_view("top", {
            "eye": [0.0, -0.1, 1.3],
            "target": [0.0, 0.0, 0.1],
            "focal_length": 18.0,
        })
        self.add_view("side", {
            "eye": [1.15, 0.0, 0.2],
            "target": [0.0, 0.0, 0.12],
        })
        self.add_view("front", {
            "eye": [0.0, 1.35, 0.25],
            "target": [0.0, 0.0, 0.12],
        })
        self.add_sensor_view("gripper_cam",
                             "/World/envs/env_0/Koch/gripper_static_1/gripper_cam")
        # self.add_viewport("top")  # 启动时默认显示 top 视角的 viewport TODO 仍有问题，仍需手动添加
        
        

        print("[INFO]: SimIsaacModel setup complete.")

    # 设置移动速度
    def set_speed(self, speed):
        self._speed = speed

    # 输入关节角度列表，单位为弧度
    def set_joint_angles(self, joint_angles):
        if isinstance(joint_angles, list):
            joint_angles = torch.tensor(joint_angles, dtype=torch.float32, device=self._sim.device)
        self._target_pos[0, :len(joint_angles)] = joint_angles
        self._robot.set_joint_position_target(self._target_pos)
        # 步进仿真
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim_dt)

    # 获取当前关节角度列表，单位为弧度
    def get_joint_angles(self, joint_angles=None):
        return self._robot.data.joint_pos[0].cpu().tolist()
    
    def add_sensor_view(self, view_name, camera_prim_path, view_params = None):
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
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
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
            print(f"[Viewport] Error: view '{view_name}' not found. Available views: {list(self._views.keys())}")
            return None

        camera_path = self._views[view_name]["camera_path"]
        window_name = f"{view_name}_Viewport"

        try:
            # 使用 viewport API 创建新的 3D viewport 窗口
            viewport_api = omni.kit.viewport.utility.get_viewport_interface()

            # 创建新的 viewport 窗口
            viewport_api.create_viewport_window(window_name)

            # 异步等待窗口创建并设置相机
            asyncio.ensure_future(self._setup_viewport_camera(window_name, camera_path, view_name))

            print(f"[Viewport] Creating viewport window '{window_name}' for camera '{camera_path}'")
            print(f"[Viewport] Window will be docked next to main Viewport once ready")

            return window_name

        except Exception as e:
            print(f"[Viewport] Failed to create viewport window: {e}")
            print(f"[Viewport] Tip: You can use switch_view('{view_name}') to change the main viewport instead")
            return None

    async def _setup_viewport_camera(self, window_name: str, camera_path: str, view_name: str):
        """异步设置 viewport 的相机并停靠窗口"""
        # 等待窗口创建完成
        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

            # 尝试获取新创建的 viewport
            new_viewport = get_viewport_from_window_name(window_name)

            if new_viewport:
                # 设置相机路径
                new_viewport.set_active_camera(camera_path)
                print(f"[Viewport] Successfully set camera '{camera_path}' for viewport '{window_name}'")

                # 停靠窗口
                await self._dock_viewport_window(window_name)
                return

        print(f"[Viewport] Warning: Could not get viewport handle for '{window_name}' after creation")

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
            print(f"[Viewport] Window '{window_name}' created but could not dock (main Viewport not found)")
        else:
            print(f"[Viewport] Could not find window '{window_name}' in workspace")
    
    # 切换稳定模式（增加阻尼，减少晃动）
    def toggle_stable_mode(self, enable):
        self._stable_mode = enable
        if enable:
            # 增加阻尼和刚度以减少晃动
            stiffness = 800.0
            damping = 80.0
        else:
            stiffness = 400.0
            damping = 40.0

        # 通过写入 actuator 的属性来调整 PD 增益
        self._robot.actuators["all_joints"].stiffness[:] = stiffness
        self._robot.actuators["all_joints"].damping[:] = damping
        print(f"[Stable] {'Enabled' if enable else 'Disabled'} (stiffness={stiffness}, damping={damping})")

    def joint_angles_to_poses(self, joint_angles, last_joint_body_name="gripper_static_1", gripper_body_name="gripper_moving_1"):
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
            'last_joint': {'pos': last_pos, 'quat': last_quat},
            'gripper':    {'pos': gripper_pos, 'quat': gripper_quat},
        }

    def get_current_poses(self):
        """
        从当前仿真状态直接读取末端位姿

        Returns
        -------
        dict  （格式同 joint_angles_to_poses）
        """
        ee_pos = self._robot.data.body_pos_w[0, self._ee_body_id].cpu().numpy()
        ee_quat = self._robot.data.body_quat_w[0, self._ee_body_id].cpu().numpy()  # [w,x,y,z]

        grip_pos = self._robot.data.body_pos_w[0, self._gripper_body_id].cpu().numpy()
        grip_quat = self._robot.data.body_quat_w[0, self._gripper_body_id].cpu().numpy()

        return {
            'last_joint': {'pos': ee_pos, 'quat': ee_quat},
            'gripper':    {'pos': grip_pos, 'quat': grip_quat},
        }

    # 移动到预设的 home 位姿（所有关节角度为 0）
    def move_to_home(self):
        pass

    # isaac sim ik 逆运动学求解 - 并插值，生成平滑轨迹
    def isaac_ik_trace(self, pos, quat = None, rot_rad = 0, steps=10):
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
            target_pos_w = self._robot.data.body_pos_w[:, self._robot_entity_cfg.body_ids[0]]
        else :
            target_pos_w = torch.as_tensor(pos, dtype=torch.float32, device=device).reshape(1, 3)


        ee_pos_w = self._robot.data.body_pos_w[:, self._robot_entity_cfg.body_ids[0]]
        ee_quat_w = self._robot.data.body_quat_w[:, self._robot_entity_cfg.body_ids[0]]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w

        if quat is None:
            target_quat_w = ee_quat_w.clone()
        else:
            target_quat_w = torch.as_tensor(quat, dtype=torch.float32, device=device).reshape(1, 4)
            quat_norm = torch.linalg.norm(target_quat_w, dim=1, keepdim=True)
            if torch.any(quat_norm < 1e-8):
                raise ValueError("Target quaternion norm must be non-zero.")
            target_quat_w = target_quat_w / quat_norm
            
        if rot_rad != 0:
            # 构造绕 Z 轴旋转的四元数
            rot_quat = torch.tensor([
                math.cos(rot_rad/2),  # w: 实部
                0,                    # x: 绕X轴分量为0
                0,                    # y: 绕Y轴分量为0  
                math.sin(rot_rad/2)   # z: 绕Z轴分量
            ], dtype=torch.float32, device=device).reshape(1, 4)
            
            # 将新旋转叠加到原朝向上
            target_quat_w = _quat_multiply(rot_quat, target_quat_w)

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)

        ik_command = torch.cat((target_pos_b, target_quat_b), dim=1)
        self._ik_controller.reset()
        self._ik_controller.set_command(ik_command)

        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._robot_entity_cfg.joint_ids]
        joint_pos = self._robot.data.joint_pos[:, self._robot_entity_cfg.joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        current_joint_pos = self._robot.data.joint_pos[0].clone()
        target_joint_pos = current_joint_pos.clone()
        target_joint_pos[self._robot_entity_cfg.joint_ids] = joint_pos_des[0]

        trajectory = []
        for i in range(steps):
            alpha = 1.0 if steps == 1 else i / (steps - 1)
            waypoint = current_joint_pos + alpha * (target_joint_pos - current_joint_pos)
            trajectory.append(waypoint.detach().cpu().tolist())

        return trajectory

    # ik 逆运动学控制器接口，输入末端位姿（位置 + 旋转），输出关节角度
    def joint_poses_to_angles_to(self, target_poses)->dict:
        """
        逆运动学：根据末端位姿计算关节角度。

        Parameters
        ----------
        target_poses : dict
            目标位姿，格式同 joint_angles_to_poses 输出。

        Returns
        -------
        dict
            {
                'success': bool,  # 是否成功找到解
                'joint_angles': list[float],  # 关节角度列表（弧度），长度与 model.nq 一致
            }
        """
        target_pos = target_poses['last_joint']['pos']
        target_quat = target_poses['last_joint']['quat']

        # 使用数值雅可比迭代 IK
        q = self._robot.data.joint_pos[0].clone().cpu().float()
        target = torch.tensor(target_pos, dtype=torch.float32)
        num_arm_joints = 5
        eps = 1e-4
        max_iter = 200
        tol = 0.001
        damping = 0.01
        lr = 0.5

        for iteration in range(max_iter):
            T = _make_fk(q.tolist(), up_to_joint=5)
            ee_pos = T[:3, 3]
            error = target - ee_pos

            if error.norm().item() < tol:
                return {
                    'success': True,
                    'joint_angles': q.tolist(),
                }

            # 数值雅可比 (3 x 5)
            J = torch.zeros(3, num_arm_joints)
            for j in range(num_arm_joints):
                q_pert = q.clone()
                q_pert[j] += eps
                T_pert = _make_fk(q_pert.tolist(), up_to_joint=5)
                J[:, j] = (T_pert[:3, 3] - ee_pos) / eps

            # 阻尼最小二乘: dq = J^T (J J^T + λI)^{-1} error
            JJT = J @ J.T + damping * torch.eye(3)
            dq = J.T @ torch.linalg.solve(JJT, error)
            q[:num_arm_joints] += lr * dq

        return {
            'success': False,
            'joint_angles': q.tolist(),
        }

    def contact_sensor_infor(self):
        # print information from the sensors
        print("-------------------------------")
        print(self._scene["gripper_move_contact_cfg"])
        print("Received force matrix of: ", self._scene["gripper_move_contact_cfg"].data.force_matrix_w)
        print("Received contact force of: ", self._scene["gripper_move_contact_cfg"].data.net_forces_w)
        print("-------------------------------")
        print(self._scene["gripper_static_contact_cfg"])
        print("Received force matrix of: ", self._scene["gripper_static_contact_cfg"].data.force_matrix_w)
        print("Received contact force of: ", self._scene["gripper_static_contact_cfg"].data.net_forces_w)
        print("-------------------------------")
        print(self._scene["cube_contact_forces"])
        print("Received force matrix of: ", self._scene["cube_contact_forces"].data.force_matrix_w)
        print("Received contact force of: ", self._scene["cube_contact_forces"].data.net_forces_w)
        print("-------------------------------")

        
    def step(self):
        """执行一步仿真"""
        self._robot.set_joint_position_target(self._target_pos)
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim_dt)
        if DEBUG_MODE:
            self.contact_sensor_infor()

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
        
        
    def grasp_open(self):
        """打开夹爪"""
        self._target_pos[0, 5] = GRIPPER_OPEN
        self._robot.set_joint_position_target(self._target_pos)

    def close_gripper(self):
        """关闭夹爪"""
        self._target_pos[0, 5] = GRIPPER_CLOSED
        self._robot.set_joint_position_target(self._target_pos)

    def gripper_state(self)->float:
        """返回夹爪电机的角度"""
        return self._robot.data.joint_pos[0, 5].cpu().item()

    def is_grasping(self, threshold=0.02)->bool:
        """根据夹爪当前状态判断是否正在夹持物体，threshold 是夹爪闭合的角度阈值（弧度）"""
        gripper_angle = self.gripper_state()
        # If gripper is more closed than threshold from fully open, consider it grasping
        return abs(gripper_angle - GRIPPER_OPEN) > threshold

    def get_object_position(self)->np.ndarray:
        """获取场景中物体的当前位置

        Returns:
            np.ndarray: 物体的3D位置 [x, y, z]，如果场景中没有物体则返回None
        """
        if "cube" not in self._scene.keys():
            print("[Object] No cube in scene")
            return None

        cube = self._scene["cube"]
        return cube.data.root_pos_w[0].cpu().numpy()

    def randomize_object(self, position=None)->dict:
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
        yaw = random.uniform(0, 2 * math.pi)
        qw = math.cos(yaw / 2)
        qx, qy = 0.0, 0.0
        qz = math.sin(yaw / 2)
        if DEBUG_MODE :
            # 固定位置和朝向用于调试
            x, y, z = 0.0, 0.15, OBJ_Z
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            position = (x, y, z)
            

        # Write to simulation
        pose = torch.tensor([[x, y, z, qw, qx, qy, qz]], dtype=torch.float32, device=cube.device)
        pose[:, :3] += self._scene.env_origins
        cube.write_root_pose_to_sim(pose)

        # Clear velocity to prevent residual motion
        zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=cube.device)
        cube.write_root_velocity_to_sim(zero_vel)

        print(f"[Object] Randomized cube at ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # 根据 rot 算出旋转角度
        euler_x, euler_y, euler_z = euler_from_quaternion(qw, qx, qy, qz)
        print(f"[Object] Cube quaternion: (w={qw:.3f}, x={qx:.3f}, y={qy:.3f}, z={qz:.3f})")
        print(f"[Object] Cube orientation (Euler angles): ({euler_x}°, {euler_y}°, {euler_z}°)")

        return {
            'name': 'cube',
            'position': position,
            'orientation': (qw, qx, qy, qz),
            'euler_angles': (euler_x, euler_y, euler_z),
        }

    def remove_random_object(self):
        """移除之前添加的随机物体"""
        if not self._random_objects:
            print("[Object] No random objects to remove")
            return

        stage = get_current_stage()
        for obj_info in self._random_objects:
            prim_path = obj_info['prim_path']
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
                visual_material=self._obj_visual_material
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x, y, z),
                rot=(qw, qx, qy, qz),
            ),
        )

        # Spawn the object
        obj_cfg.spawn.func(prim_path, obj_cfg.spawn, translation=(x, y, z), orientation=(qw, qx, qy, qz))

        print(f"[Object] Added: {name} at ({x:.3f}, {y:.3f}, {z:.3f})")
        # 根据 rot 算出旋转角度
        euler_x, euler_y, euler_z = euler_from_quaternion(qw, qx, qy, qz)

        return {
            'name': name,
            'prim_path': prim_path,
            'position': position,
            'size': size,
            'orientation': (qw, qx, qy, qz),
            'euler_angles': (euler_x, euler_y, euler_z),
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
    if axis == 'x':
        T[1, 1], T[1, 2] = c, -s
        T[2, 1], T[2, 2] = s,  c
    elif axis == 'y':
        T[0, 0], T[0, 2] =  c, s
        T[2, 0], T[2, 2] = -s, c
    else:  # z
        T[0, 0], T[0, 1] = c, -s
        T[1, 0], T[1, 1] = s,  c
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
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
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
    sim.toggle_stable_mode(True)

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
    print("    V          - Cycle viewpoints (main -> top -> side -> front -> gripper_cam -> close_up -> main)")
    print("=" * 50)
    print(f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})")

    # 主循环
    while simulation_app.is_running():
        '''
        Joint mode:
            UP/DOWN    - Select joint
            LEFT/RIGHT - Decrease/Increase joint angle
            R         - Reset all joints to zero
        '''

        # 选择关节
        if kb.on_press("UP"):
            current_joint_idx = (current_joint_idx - 1) % len(JOINT_NAMES)
            print(f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})")
        if kb.on_press("DOWN"):
            current_joint_idx = (current_joint_idx + 1) % len(JOINT_NAMES)
            print(f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})")

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
            ee = poses['last_joint']
            angles = sim.get_joint_angles()
            angles_str = ", ".join(f"{JOINT_NAMES[i]}: {angles[i]:.4f}" for i in range(len(JOINT_NAMES)))
            print(f"[EE] pos: ({ee['pos'][0]:.4f}, {ee['pos'][1]:.4f}, {ee['pos'][2]:.4f})  "
                  f"[Joints] {angles_str}")

    # 清理
    kb.stop()
    # sim.close()

def data_produce():
    """Main loop: iterate over episodes, generate trajectories, record data.
    Overall flow per episode:
        1. Reset robot to home position
        2. Randomize object position/orientation on table
        3. Plan grasp trajectory (IK for each waypoint)
        4. Execute trajectory step-by-step:
           a. Set joint position target
           b. Step simulation
           c. Update cameras
           d. Record step data to HDF5
        5. Check grasp success
        6. Save episode file
    """
    num_episodes = NUM_EXPISODES
    output_dir = "datasets/grasp_v1"

    # 创建仿真器实例，加载 URDF 模型
    sim = SimIsaacModel(URDF_PATH)

    # 获取相机传感器
    camera_gripper: Camera = sim._scene["gripper_cam"]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0

    print(f"\n[INFO]: Starting data generation: {num_episodes} episodes → {output_dir}")

    for ep_idx in range(num_episodes):
        print(f"\n[Episode {ep_idx + 1}/{num_episodes}]")

        # 1. Reset robot to home position
        sim.reset()
        sim.grasp_open()

        # Settle the scene
        for _ in range(5):
            sim.step()

        # 2. Randomize object position
        obj_info = sim.randomize_object()
        object_rot = obj_info['orientation']
        object_euler = obj_info['euler_angles']
        object_pos = obj_info['position']

        # Settle after object spawn
        for _ in range(5):
            sim.step()

        # 3. Create HDF5 file for this episode
        filepath = os.path.join(output_dir, f"episode_{ep_idx:06d}.hdf5")
        f = h5py.File(filepath, "w")

        max_steps = STEPS_PER_PHASE * 7
        f.create_dataset("action", shape=(0, 6), maxshape=(max_steps, 6), dtype="float32")
        f.create_dataset("observation/images/gripper", shape=(0, CAM_HEIGHT, CAM_WIDTH, 3),
                         maxshape=(max_steps, CAM_HEIGHT, CAM_WIDTH, 3), dtype="uint8")
        f.create_dataset("observation/state", shape=(0, 6), maxshape=(max_steps, 6), dtype="float32")
        f.create_dataset("observation/gripper", shape=(0, 1), maxshape=(max_steps, 1), dtype="float32")
        f.create_dataset("object_pos", shape=(0, 3), maxshape=(max_steps, 3), dtype="float32")

        f.attrs["fps"] = 30
        f.attrs["sim_dt"] = sim._sim_dt
        f.attrs["success"] = False

        try:
            # Store initial object position for grasp detection
            initial_obj_pos = np.array(object_pos, dtype=np.float32)

            # 4.5. plan and execute
            fin_flag = False
            missions = ["move_up", "move_down", "grasp_close", "lift_up", "hold"]
            while not fin_flag:
                # 遍历 missions 列表
                for mission in missions:
                    print(f"  mission: {mission}")
                    phases = plan_grasp_trajectory(sim, mission, object_pos, object_euler, PLACE_POS)
                    # 5. Execute trajectory and record data
                    for phase in phases:
                        print(f"  Phase: {phase['name']}")
                        for waypoint in phase["waypoints"]:
                            # Set joint targets
                            target = torch.tensor(waypoint, dtype=torch.float32, device=sim._sim.device)
                            target[5] = phase["gripper"]
                            sim._target_pos[0] = target
                            sim.step()
                            # Record step data
                            record_step(f, sim, camera_gripper, target, phase["gripper"])
                    # time.sleep(3)

                
                fin_flag = True
            grasp_status = check_grasp_success_dual(sim, initial_obj_pos)


            print(f"    Grasp Detection:")
            print(f"      - Gripper closed: {grasp_status['gripper_closed']} (angle: {grasp_status['gripper_angle']:.3f})")
            print(f"      - Object lifted: {grasp_status['object_lifted']} (Δh: {grasp_status['height_delta']:.3f}m)")
            print(f"      - Object position: [{grasp_status['object_pos'][0]:.3f}, {grasp_status['object_pos'][1]:.3f}, {grasp_status['object_pos'][2]:.3f}]")
            print(f"      - Grasp SUCCESS: {grasp_status['grasped']}")

            # 6. Check final placement success
            success = check_grasp_success(sim, PLACE_POS)

        except Exception as exc:
            print(f"  [Episode {ep_idx + 1}] aborted: {exc}")
            success = False
        finally:
            f.attrs["success"] = success
            f.attrs["num_steps"] = f["action"].shape[0]
            f.close()

        if success:
            success_count += 1
        print(f"  Result: {'SUCCESS' if success else 'FAIL'} ({success_count}/{ep_idx + 1})")

        if not simulation_app.is_running():
            break

    print(f"\n{'=' * 50}")
    print(f"Data generation complete: {num_episodes} episodes")
    print(f"Success rate: {success_count}/{num_episodes}")
    print(f"Output: {output_dir}")

    # sim.close()


# 按照规则
def plan_grasp_trajectory(sim: SimIsaacModel, mission: str, object_pos: tuple, object_euler: tuple, place_pos: tuple) -> list[dict]:
    """Plan a full pick-and-place trajectory as a sequence of phases.

    Optimized grasp logic:
    1. Move to pre-grasp position above object with gripper open
    2. Descend to grasp position while keeping gripper open
    3. Close gripper to grasp object
    4. Lift object up
    5. Return to home position while maintaining grasp
    6. Move to place position
    7. Descend to place
    8. Release gripper
    9. Return to home position

    Each phase is planned based on the final waypoint of the previous phase,
    ensuring continuous and deterministic trajectory planning without relying
    on real-time servo angles.
    
    
    预先定义好的是这几个任务， TODO 后续改成 vla 规划出来 phases 后面的改为参数形式
    missions = ["move_up", "move_down", "grasp_close", "lift_up", "move_home"]
    """
    phases = []

    grasp_height = max(object_pos[2] , OBJ_Z)
    place_height = max(place_pos[2], OBJ_Z)

    # Get home position for return trajectory
    home_joints = sim._robot.data.default_joint_pos[0, :6].cpu().tolist()
    
    # 根据 object_rot 确定 物体的旋转角度是多少
    
    yaw = object_euler[2] 
    

    if mission == "move_up":
        # Phase 1: Approach (move to pre-grasp position above object)
        pre_grasp_pos = (object_pos[0] + GRIPPER_OFFSET * math.cos(yaw), object_pos[1] + GRIPPER_OFFSET * math.sin(-yaw), grasp_height + PRE_GRASP_HEIGHT_OFFSET + OBJ_H * 2)
        # 进行一个旋转 yaw 角度的偏移，确保和物体的朝向一致，增加抓取成功率
        
        
        pre_grasp_traj = sim.isaac_ik_trace(pre_grasp_pos, rot_rad=yaw, steps=STEPS_PER_PHASE)
        phases.append({
            "name": "approach",
            "waypoints": pre_grasp_traj,
            "gripper": GRIPPER_OPEN,
        })
    elif mission == "grasp_open":
        # 只打开夹爪 位置不改变
        # 获取 关节角度值
        joints_at_grasp = sim._robot.data.joint_pos[0, :6].cpu().tolist()  # Get first 6 joints (exclude gripper)
        phases.append({
            "name": "grasp_open",
            "waypoints": [joints_at_grasp for _ in range(STEPS_PER_PHASE)],
            "gripper": GRIPPER_OPEN,
        })
    elif mission == "move_down":
        # Phase 2: Descend to grasp position (gripper remains open)
        grasp_pos = (object_pos[0]+ GRIPPER_OFFSET * math.cos(yaw), object_pos[1] + GRIPPER_OFFSET * math.sin(-yaw), OBJ_H)
        grasp_traj = sim.isaac_ik_trace(grasp_pos, rot_rad=yaw, steps=STEPS_PER_PHASE)
        phases.append({
            "name": "descend",
            "waypoints": grasp_traj,
            "gripper": GRIPPER_OPEN,
        })
    elif mission == "grasp_close":
        # Phase 3: Close gripper to grasp object
        # Use the final waypoint from Phase 2 (descend)
        joints_at_grasp = sim._robot.data.joint_pos[0, :6].cpu().tolist()
        phases.append({
            "name": "close_gripper",
            "waypoints": [joints_at_grasp for _ in range(STEPS_PER_PHASE)],
            "gripper": GRIPPER_GRASP,
        })
    elif mission == "lift_up":
        # Phase 4: Lift object up
        # Based on the final waypoint of Phase 3 (which is same as Phase 2's end position)
        lift_pos = (object_pos[0] , object_pos[1] , max(LIFT_HEIGHT, grasp_height + PRE_GRASP_HEIGHT_OFFSET))
        lift_traj = sim.isaac_ik_trace(lift_pos, steps=STEPS_PER_PHASE)
        phases.append({
            "name": "lift",
            "waypoints": lift_traj,
            "gripper": GRIPPER_GRASP,
        })
    elif mission == "hold":
        joints_at_grasp = sim._robot.data.joint_pos[0, :6].cpu().tolist()
        phases.append({
            "name": "hold",
            "waypoints": [joints_at_grasp for _ in range(STEPS_PER_PHASE*2)],
            "gripper": GRIPPER_GRASP,
        })
    return phases


def record_step(f: h5py.File, sim: SimIsaacModel, camera_gripper: Camera, action: torch.Tensor, gripper_val: float):
    """Record one timestep of data into the HDF5 file."""
    # Read current state
    joint_state = sim._robot.data.joint_pos[0, :6].cpu().numpy()
    gripper_state = np.array([gripper_val], dtype=np.float32)

    # Get object position from scene cube
    cube = sim._scene["cube"]
    object_pos = cube.data.root_pos_w[0].cpu().numpy()

    # Read camera image
    gripper_rgb = camera_gripper.data.output["rgb"][0, ..., :3].cpu().numpy()

    # Append to HDF5
    for key, data in [
        ("action", action.cpu().numpy().reshape(1, 6)),
        ("observation/state", joint_state.reshape(1, 6)),
        ("observation/gripper", gripper_state.reshape(1, 1)),
        ("observation/images/gripper", gripper_rgb[np.newaxis]),
        ("object_pos", object_pos.reshape(1, 3)),
    ]:
        ds = f[key]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = data


def check_grasp_success_dual(sim: SimIsaacModel, initial_obj_pos: np.ndarray,
                             gripper_closed_threshold: float = 0.02,
                             height_threshold: float = 0.05) -> dict:
    """Check if object is grasped using dual criteria:

    1. Gripper state: Check if gripper is closed (indicating contact with object)
    2. Object position: Check if object has been lifted from initial position

    Args:
        sim: Simulation model
        initial_obj_pos: Initial object position before grasp attempt
        gripper_closed_threshold: Minimum gripper closure angle to consider grasping
        height_threshold: Minimum height increase to confirm object is lifted

    Returns:
        dict with keys:
            - 'grasped': bool, True if both criteria are met
            - 'gripper_closed': bool, gripper closure state
            - 'object_lifted': bool, object lift state
            - 'gripper_angle': float, current gripper angle
            - 'object_pos': np.ndarray, current object position
            - 'height_delta': float, height change from initial position
    """
    # Criterion 1: Check gripper state
    gripper_angle = sim.gripper_state()
    gripper_closed = abs(gripper_angle - GRIPPER_OPEN) > gripper_closed_threshold

    # Criterion 2: Check object position
    cube = sim._scene["cube"]
    current_obj_pos = cube.data.root_pos_w[0].cpu().numpy()
    height_delta = current_obj_pos[2] - initial_obj_pos[2]
    object_lifted = height_delta > height_threshold

    # Both criteria must be met for successful grasp
    grasped = gripper_closed and object_lifted

    return {
        'grasped': grasped,
        'gripper_closed': gripper_closed,
        'object_lifted': object_lifted,
        'gripper_angle': gripper_angle,
        'object_pos': current_obj_pos,
        'height_delta': height_delta,
    }


def check_grasp_success(sim: SimIsaacModel, place_pos: tuple, threshold: float = 0.05) -> bool:
    """Check if the object was successfully placed near the target position."""
    cube = sim._scene["cube"]
    obj_pos = cube.data.root_pos_w[0].cpu().numpy()
    place_pos_np = np.array(place_pos, dtype=np.float32)
    distance = np.linalg.norm(obj_pos - place_pos_np)

    return bool(distance < threshold and obj_pos[2] >= OBJ_Z * 0.8)
    

if __name__ == "__main__":
    # demo_control()
    data_produce()
    # 最后关闭
    print("\n[INFO]: Simulation finished, closing application.")
    simulation_app.close()
    print("[INFO]: Application closed successfully.")
