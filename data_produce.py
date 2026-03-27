"""Koch robot arm - scripted grasp data generation.

Generates pick-and-place episodes with two cameras (top + wrist).
Each episode: randomize object → IK to pre-grasp → descend → close gripper → lift → move → release.

Usage:
    ./IsaacLab/isaaclab.sh -p data_produce.py --num_episodes 1000 --output_dir datasets/grasp_v1
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Koch arm grasp data generation")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to generate")
parser.add_argument("--output_dir", type=str, default="datasets/grasp_v1", help="Output directory for HDF5 files")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-launch imports (must be after AppLauncher) ---

import math
import random

import h5py
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass


# =============================================================================
# Section 1: Configuration Constants
# =============================================================================

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "koch.urdf")

# Table dimensions
TABLE_HEIGHT = 0.05
TABLE_SIZE = (0.3, 0.4, TABLE_HEIGHT)
TABLE_POS = (0.2, 0.0, TABLE_HEIGHT / 2)  # in front of the robot base

# Object randomization ranges (on table surface)
OBJ_X_RANGE = (0.10, 0.30)   # forward from robot base
OBJ_Y_RANGE = (-0.15, 0.15)  # left-right
OBJ_SIZE_RANGE = (0.02, 0.04)  # cube side length

# Place target (fixed position to drop the object)
PLACE_POS = (0.15, 0.15, TABLE_HEIGHT + 0.05)

# Grasp approach parameters
PRE_GRASP_HEIGHT_OFFSET = 0.08  # meters above the object for pre-grasp
LIFT_HEIGHT = 0.15  # meters above table after grasping

# Gripper joint values
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -0.6  # TODO: calibrate to actual Koch gripper range

# Trajectory interpolation
STEPS_PER_PHASE = 30  # simulation steps per motion phase

# Camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Data recording FPS
RECORD_FPS = 30


# =============================================================================
# Section 2: Robot Configuration (reuse from isaac-sim.py)
# =============================================================================

KOCH_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
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
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)


# =============================================================================
# Section 3: Scene Configuration
# =============================================================================

@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Scene with Koch arm + table + object + two cameras."""

    # --- Environment basics ---
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # --- Koch arm ---
    koch = KOCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Koch")

    # --- Table (fixed rigid body) ---
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # --- Target object (dynamic cuboid, will be randomized) ---
    # TODO: 用 RigidObjectCfg 替代 AssetBaseCfg 以支持动态物理
    # TODO: 添加随机颜色材质
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_POS[0], 0.0, TABLE_HEIGHT + 0.015),
        ),
    )

    # --- Top camera (fixed, looking down at the table) ---
    camera_top = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraTop",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 0.6),       # above the table center
            rot=(0.0, 0.0, 1.0, 0.0),  # looking straight down (180° around Y)
            convention="world",
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        data_types=["rgb"],
        update_period=0,
    )

    # --- Wrist camera (mounted on link4_1, follows end-effector) ---
    camera_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Koch/link4_1/CameraWrist",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.02),       # slightly above link4
            rot=(1.0, 0.0, 0.0, 0.0),   # TODO: 调整朝向使其朝夹爪方向看
            convention="ros",
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        data_types=["rgb"],
        update_period=0,
    )


# =============================================================================
# Section 4: Inverse Kinematics Helper
# =============================================================================

def solve_ik(robot: Articulation, target_pos_world: tuple, target_orient: tuple | None = None) -> torch.Tensor:
    """Solve IK for the Koch arm to reach a target position in world frame.

    Args:
        robot: The Koch Articulation instance.
        target_pos_world: (x, y, z) desired end-effector position.
        target_orient: (qw, qx, qy, qz) desired orientation, or None for default.

    Returns:
        joint_positions: (6,) tensor of target joint angles.

    TODO: 实现逆运动学求解
    方案选择 (按优先级):
      1. isaaclab.controllers.DifferentialIKController
         - 基于雅可比矩阵的迭代求解
         - 需要 end-effector frame name
         - 参考: IsaacLab/scripts/tutorials/05_controllers/run_diff_ik.py
      2. 解析解 (analytic IK)
         - Koch 是 5+1 DOF, 可能存在解析解
         - 需要手动推导 DH 参数
      3. 简单几何方法
         - 对于桌面抓取, 末端垂直向下时可简化为 2D 平面问题
         - joint1 控制旋转, joint2-4 控制伸展高度

    临时方案: 返回预设关节角度 (调试用)
    """
    # TODO: replace with actual IK solver
    return robot.data.default_joint_pos[0].clone()


# =============================================================================
# Section 5: Trajectory Generation
# =============================================================================

def interpolate_joints(start: torch.Tensor, end: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """Generate linearly interpolated joint trajectory between two configurations.

    Args:
        start: (num_joints,) starting joint positions.
        end: (num_joints,) ending joint positions.
        steps: Number of interpolation steps.

    Returns:
        List of (num_joints,) tensors, length = steps.

    TODO: 可替换为样条插值 (cubic spline) 使轨迹更平滑
    """
    trajectory = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        trajectory.append(start + alpha * (end - start))
    return trajectory


def plan_grasp_trajectory(
    robot: Articulation,
    object_pos: tuple,
    place_pos: tuple,
) -> list[dict]:
    """Plan a full pick-and-place trajectory as a sequence of phases.

    Args:
        robot: The Koch Articulation instance.
        object_pos: (x, y, z) world position of the object center.
        place_pos: (x, y, z) world position of the place target.

    Returns:
        List of phase dicts, each containing:
            - "name": str, phase name for logging
            - "waypoints": list of (num_joints,) tensors
            - "gripper": float, gripper target for this phase

    TODO: 实现完整轨迹规划, 以下为各阶段伪代码
    """
    phases = []
    home_joints = robot.data.default_joint_pos[0].clone()

    # --- Phase 1: Home → Pre-grasp (above object) ---
    pre_grasp_pos = (object_pos[0], object_pos[1], object_pos[2] + PRE_GRASP_HEIGHT_OFFSET)
    pre_grasp_joints = solve_ik(robot, pre_grasp_pos)
    phases.append({
        "name": "approach",
        "waypoints": interpolate_joints(home_joints, pre_grasp_joints, STEPS_PER_PHASE),
        "gripper": GRIPPER_OPEN,
    })

    # --- Phase 2: Pre-grasp → Grasp (descend to object) ---
    grasp_pos = (object_pos[0], object_pos[1], object_pos[2])
    grasp_joints = solve_ik(robot, grasp_pos)
    phases.append({
        "name": "descend",
        "waypoints": interpolate_joints(pre_grasp_joints, grasp_joints, STEPS_PER_PHASE),
        "gripper": GRIPPER_OPEN,
    })

    # --- Phase 3: Close gripper ---
    phases.append({
        "name": "close_gripper",
        "waypoints": [grasp_joints.clone() for _ in range(STEPS_PER_PHASE)],
        "gripper": GRIPPER_CLOSED,
    })

    # --- Phase 4: Lift object ---
    lift_pos = (object_pos[0], object_pos[1], TABLE_HEIGHT + LIFT_HEIGHT)
    lift_joints = solve_ik(robot, lift_pos)
    phases.append({
        "name": "lift",
        "waypoints": interpolate_joints(grasp_joints, lift_joints, STEPS_PER_PHASE),
        "gripper": GRIPPER_CLOSED,
    })

    # --- Phase 5: Move to place position ---
    pre_place_pos = (place_pos[0], place_pos[1], TABLE_HEIGHT + LIFT_HEIGHT)
    pre_place_joints = solve_ik(robot, pre_place_pos)
    phases.append({
        "name": "move_to_place",
        "waypoints": interpolate_joints(lift_joints, pre_place_joints, STEPS_PER_PHASE),
        "gripper": GRIPPER_CLOSED,
    })

    # --- Phase 6: Descend to place ---
    place_joints = solve_ik(robot, place_pos)
    phases.append({
        "name": "place_descend",
        "waypoints": interpolate_joints(pre_place_joints, place_joints, STEPS_PER_PHASE),
        "gripper": GRIPPER_CLOSED,
    })

    # --- Phase 7: Open gripper (release) ---
    phases.append({
        "name": "release",
        "waypoints": [place_joints.clone() for _ in range(STEPS_PER_PHASE)],
        "gripper": GRIPPER_OPEN,
    })

    return phases


# =============================================================================
# Section 6: Object Randomization
# =============================================================================

def randomize_object(cube, env_origins: torch.Tensor) -> tuple:
    """Randomize the target object's position and orientation on the table.

    Args:
        cube: The RigidObject representing the graspable cube.
        env_origins: (num_envs, 3) environment origin positions.

    Returns:
        object_pos: (x, y, z) world position of the object.

    TODO: 实现随机化逻辑
    """
    # --- 随机位置 ---
    x = random.uniform(*OBJ_X_RANGE)
    y = random.uniform(*OBJ_Y_RANGE)
    obj_half_size = 0.015  # half of default 0.03 cube
    z = TABLE_HEIGHT + obj_half_size
    object_pos = (x, y, z)

    # --- 随机朝向 (仅 yaw) ---
    yaw = random.uniform(0, 2 * math.pi)
    qw = math.cos(yaw / 2)
    qx, qy = 0.0, 0.0
    qz = math.sin(yaw / 2)

    # --- 写入仿真 ---
    pose = torch.tensor([[x, y, z, qw, qx, qy, qz]], dtype=torch.float32, device=cube.device)
    pose[:, :3] += env_origins
    cube.write_root_pose_to_sim(pose)
    # 清零速度，防止残留上一轮运动
    zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=cube.device)
    cube.write_root_velocity_to_sim(zero_vel)

    return object_pos


# =============================================================================
# Section 7: Data Recording
# =============================================================================

def create_episode_file(output_dir: str, episode_idx: int) -> h5py.File:
    """Create an HDF5 file for one episode.

    Args:
        output_dir: Base output directory.
        episode_idx: Episode index number.

    Returns:
        Open h5py.File handle with pre-created datasets.

    File structure:
        episode_XXXXXX.hdf5
        ├── action                    (T, 6)   float32  - target joint positions
        ├── observation/
        │   ├── images/
        │   │   ├── top               (T, 480, 640, 3) uint8
        │   │   └── wrist             (T, 480, 640, 3) uint8
        │   ├── state                 (T, 6)   float32  - current joint positions
        │   └── gripper               (T, 1)   float32  - gripper state
        ├── object_pos                (T, 3)   float32  - object world position
        └── attrs:
              fps, sim_dt, success, num_steps
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"episode_{episode_idx:06d}.hdf5")
    f = h5py.File(filepath, "w")

    max_steps = STEPS_PER_PHASE * 7  # 7 phases
    f.create_dataset("action",                    shape=(0, 6), maxshape=(max_steps, 6), dtype="float32")
    f.create_dataset("observation/images/top",     shape=(0, CAM_HEIGHT, CAM_WIDTH, 3), maxshape=(max_steps, CAM_HEIGHT, CAM_WIDTH, 3), dtype="uint8")
    f.create_dataset("observation/images/wrist",   shape=(0, CAM_HEIGHT, CAM_WIDTH, 3), maxshape=(max_steps, CAM_HEIGHT, CAM_WIDTH, 3), dtype="uint8")
    f.create_dataset("observation/state",          shape=(0, 6), maxshape=(max_steps, 6), dtype="float32")
    f.create_dataset("observation/gripper",        shape=(0, 1), maxshape=(max_steps, 1), dtype="float32")
    f.create_dataset("object_pos",                 shape=(0, 3), maxshape=(max_steps, 3), dtype="float32")

    f.attrs["fps"] = RECORD_FPS
    f.attrs["success"] = False
    return f


def record_step(
    f: h5py.File,
    robot: Articulation,
    cube,
    camera_top: Camera,
    camera_wrist: Camera,
    action: torch.Tensor,
    gripper_val: float,
):
    """Record one timestep of data into the HDF5 file.

    Args:
        f: Open HDF5 file.
        robot: Koch arm articulation.
        cube: Target object rigid body.
        camera_top: Top-down camera sensor.
        camera_wrist: Wrist-mounted camera sensor.
        action: (6,) target joint positions sent this step.
        gripper_val: Gripper target value this step.

    TODO: 实现数据写入
    """
    # --- 读取当前状态 ---
    joint_state = robot.data.joint_pos[0, :6].cpu().numpy()          # (6,)
    gripper_state = np.array([gripper_val], dtype=np.float32)         # (1,)
    object_pos = cube.data.root_pos_w[0].cpu().numpy()                # (3,)

    # --- 读取相机图像 ---
    top_rgb = camera_top.data.output["rgb"][0, ..., :3].cpu().numpy()       # (H, W, 3) uint8
    wrist_rgb = camera_wrist.data.output["rgb"][0, ..., :3].cpu().numpy()

    # --- 追加写入 HDF5 (resize + write last row) ---
    for key, data in [
        ("action", action.cpu().numpy().reshape(1, 6)),
        ("observation/state", joint_state.reshape(1, 6)),
        ("observation/gripper", gripper_state.reshape(1, 1)),
        ("observation/images/top", top_rgb[np.newaxis]),
        ("observation/images/wrist", wrist_rgb[np.newaxis]),
        ("object_pos", object_pos.reshape(1, 3)),
    ]:
        ds = f[key]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = data


# =============================================================================
# Section 8: Grasp Success Detection
# =============================================================================

def check_grasp_success(cube, place_pos: tuple, threshold: float = 0.05) -> bool:
    """Check if the object was successfully placed near the target position.

    Args:
        cube: Target object rigid body.
        place_pos: (x, y, z) intended place position.
        threshold: Maximum distance (meters) to count as success.

    Returns:
        True if object is within threshold of place_pos.

    TODO: 实现成功检测
    """
    cube_pos = cube.data.root_pos_w[0].cpu().numpy()
    distance = np.linalg.norm(cube_pos - np.array(place_pos))
    return bool(distance < threshold)


# =============================================================================
# Section 9: Main Data Generation Loop
# =============================================================================

def run_data_generation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
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

    Args:
        sim: Isaac Sim simulation context.
        scene: Interactive scene containing all assets and sensors.
    """
    robot: Articulation = scene["koch"]
    cube = scene["cube"]
    camera_top: Camera = scene["camera_top"]
    camera_wrist: Camera = scene["camera_wrist"]
    sim_dt = sim.get_physics_dt()

    os.makedirs(args_cli.output_dir, exist_ok=True)

    success_count = 0

    for ep_idx in range(args_cli.num_episodes):
        print(f"\n[Episode {ep_idx + 1}/{args_cli.num_episodes}]")

        # --- Step 1: Reset robot ---
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )
        robot.reset()

        # --- Step 2: Randomize object ---
        object_pos = randomize_object(cube, scene.env_origins)
        sim.step()  # let physics settle
        scene.update(sim_dt)

        # --- Step 3: Plan trajectory ---
        phases = plan_grasp_trajectory(robot, object_pos, PLACE_POS)

        # --- Step 4: Create episode HDF5 ---
        f = create_episode_file(args_cli.output_dir, ep_idx)

        # --- Step 5: Execute each phase ---
        for phase in phases:
            print(f"  Phase: {phase['name']}")
            for waypoint in phase["waypoints"]:
                # Build full target: arm joints + gripper
                target = waypoint.clone()
                target[5] = phase["gripper"]  # joint_gripper is index 5

                # Set target and step
                robot.set_joint_position_target(target.unsqueeze(0))
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)

                # Record
                record_step(f, robot, cube, camera_top, camera_wrist, target, phase["gripper"])

        # --- Step 6: Check success ---
        success = check_grasp_success(cube, PLACE_POS)
        f.attrs["success"] = success
        f.attrs["num_steps"] = f["action"].shape[0]
        f.close()

        if success:
            success_count += 1
        print(f"  Result: {'SUCCESS' if success else 'FAIL'} ({success_count}/{ep_idx + 1})")

        # --- Early exit if sim closed ---
        if not simulation_app.is_running():
            break

    print(f"\n{'=' * 50}")
    print(f"Data generation complete: {args_cli.num_episodes} episodes")
    print(f"Success rate: {success_count}/{args_cli.num_episodes}")
    print(f"Output: {args_cli.output_dir}")


# =============================================================================
# Section 10: Entry Point
# =============================================================================

def main():
    num_episodes=1000
    output_dir = "datasets/grasp_v1"
    
    args_cli.num_episodes = num_episodes
    args_cli.output_dir = output_dir
    args_cli.device = "cuda:0"  # or "cpu"
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)

    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.5, 0.5, 0.5], [0.0, 0.0, 0.15])

    scene_cfg = GraspSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Scene setup complete.")
    print(f"[INFO]: Generating {args_cli.num_episodes} episodes → {args_cli.output_dir}")

    run_data_generation(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
