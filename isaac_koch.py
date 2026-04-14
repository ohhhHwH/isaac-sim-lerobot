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
import isaacsim.core.api
from isaacsim.core.api import World
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from isaaclab.sim.utils.stage import get_current_stage


# --- 常量配置 ---
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "koch.urdf")
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint_gripper"]
ANGLE_STEP = 0.05  # 每次按键旋转弧度 (~2.9°)

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

    # --- 机械臂配置 ---
    KOCH_CFG = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_PATH,
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
                joint_names_expr=[".*"], # 匹配所有关节
                effort_limit_sim=100.0, # 仿真中的力矩限制，过高可能导致不稳定，过低可能无法驱动机械臂
                velocity_limit_sim=100.0, # 仿真中的速度限制，过高可能导致不稳定，过低可能导致响应变慢
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )

    # 初始化
    def __init__(self, urdf_path):
        self._urdf_path = urdf_path
        self._speed = 1.0
        self._views = {}  # name -> CameraCfg
        self._stable_mode = False

        # 更新 URDF 路径
        self.KOCH_CFG = SimIsaacModel.KOCH_CFG.replace(
            spawn=sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                fix_base=True,
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=400.0,
                        damping=40.0,
                    ),
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
            ),
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
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            )
            koch = self.KOCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Koch")

            # 挂载在 gripper_static_1 上的相机传感器，严格参考 MuJoCo:
            # <camera name="gripper_cam" pos="0 0.08 0" xyaxes="1 0 0 0 0.8 -0.6"/>
            # 其中 xyaxes 对应旋转矩阵列向量:
            # x=(1,0,0), y=(0,0.8,-0.6), z=x×y=(0,0.6,0.8)
            # 等价为绕 x 轴旋转约 -36.87°，四元数 [w,x,y,z] ≈ [0.948683, -0.316228, 0, 0]
            gripper_cam = CameraCfg(
                # 相机在 USD stage 中的 prim 路径（挂在 gripper_static_1 下）
                prim_path="{ENV_REGEX_NS}/Koch/gripper_static_1/gripper_cam",
                update_period=0.1, # 传感器输出周期（秒）：每 0.1s 输出一次数据
                height=480, # 输出图像分辨率（像素）
                width=640,
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
                    pos=(0.0, 0.2, 0.0), # 平移偏移（x, y, z），单位为场景距离（通常米）
                    # 旋转偏移：四元数 (w, x, y, z)
                    # 注意：四元数方向和符号需要与场景其他部分一致（此处来源于 MuJoCo->Isaac 的映射）
                    rot=(0.766, -0.643, 0, 0), # 测试为正好看到夹爪 (x:-80,y:0,z:0)
                    convention="opengl", # 偏移的解释约定，例如 "world" 表示以世界/绝对参照解释，
                ),
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
    def isaac_ik_trace(self, pos, quat = None, steps=10):
        """
        参考并使用 IsaacLab 的 DifferentialIKController，求解目标末端位姿对应的关节目标，
        能够在当前关节角和目标关节角之间做线性插值，生成平滑轨迹。

        Args:
            pos: 目标末端位置，世界坐标系，shape=(3,)
            quat: 目标末端四元数 [w, x, y, z]，世界坐标系。默认保持当前末端朝向。
            steps (int, optional): 轨迹分段数. Defaults to 10.

        Returns:
            list[list[float]]: 从当前关节角到目标关节角的插值轨迹，每个元素是一组完整关节角。
        """
        steps = max(int(steps), 1)
        device = self._sim.device

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

    def step(self):
        """执行一步仿真"""
        self._robot.set_joint_position_target(self._target_pos)
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim_dt)

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

    def close(self):
        pass


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
    sim.close()

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
    num_episodes=2
    output_dir = "datasets/grasp_v1"
    
    args_cli.num_episodes = num_episodes
    args_cli.output_dir = output_dir
    args_cli.device = "cuda:0"  # or "cpu"
    
    # sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # sim = sim_utils.SimulationContext(sim_cfg)
    # sim.set_camera_view([0.5, 0.5, 0.5], [0.0, 0.0, 0.15])
    # scene_cfg = GraspSceneCfg(num_envs=1, env_spacing=2.0)
    # scene = InteractiveScene(scene_cfg)
    # sim.reset()
    
    print("[INFO]: Scene setup complete.")
    print(f"[INFO]: Generating {args_cli.num_episodes} episodes → {args_cli.output_dir}")
    
    run_data_generation(sim, scene)
    
    pass
    

if __name__ == "__main__":
    # demo_control()
    data_produce()


# 最后关闭
simulation_app.close()
