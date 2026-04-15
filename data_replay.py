"""Koch robot arm - dataset replay with custom camera views.

Replays recorded HDF5 episodes in Isaac Sim, driving the robot arm through
the saved joint trajectories. Supports adding custom camera viewpoints via
command-line arguments.

Usage:
    # Replay with default cameras (top + wrist from recording)
    ./IsaacLab/isaaclab.sh -p data_replay.py --dataset datasets/grasp_v1 --episode 0

    # Add a custom free camera view (pos_x,pos_y,pos_z,look_x,look_y,look_z)
    ./IsaacLab/isaaclab.sh -p data_replay.py --dataset datasets/grasp_v1 --episode 0 \
        --camera 0.4,0.3,0.3,0.0,0.1,0.0

    # Multiple custom cameras
    ./IsaacLab/isaaclab.sh -p data_replay.py --dataset datasets/grasp_v1 --episode 0 \
        --camera 0.4,0.3,0.3,0.0,0.1,0.0 --camera -0.3,0.2,0.4,0.0,0.1,0.0

    # Replay all episodes in sequence
    ./IsaacLab/isaaclab.sh -p data_replay.py --dataset datasets/grasp_v1 --all
    
python data_replay.py --dataset datasets/grasp_v1 --episode 0
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Koch arm dataset replay")
parser.add_argument("--dataset", type=str, default="datasets/grasp_v1", help="Dataset directory with HDF5 files")
parser.add_argument("--episode", type=int, default=3, help="Episode index to replay")
parser.add_argument("--all", action="store_true", help="Replay all episodes in sequence")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument(
    "--camera", action="append", default=None,
    help="Custom camera: pos_x,pos_y,pos_z,look_x,look_y,look_z (can repeat)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-launch imports (must be after AppLauncher) ---

import glob

import h5py
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


# =============================================================================
# Configuration
# =============================================================================

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "koch.urdf")

CAM_WIDTH = 640
CAM_HEIGHT = 480
OBJ_Z = 0.015

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
# Parse custom cameras
# =============================================================================

def parse_camera_args(camera_args: list[str] | None) -> list[dict]:
    """Parse --camera arguments into camera config dicts.

    Each arg is "pos_x,pos_y,pos_z,look_x,look_y,look_z".
    Returns list of dicts with 'pos' and 'target' keys.
    """
    if not camera_args:
        return []
    cameras = []
    for i, cam_str in enumerate(camera_args):
        values = [float(v) for v in cam_str.split(",")]
        if len(values) != 6:
            raise ValueError(
                f"--camera expects 6 values (pos_x,pos_y,pos_z,look_x,look_y,look_z), got {len(values)}"
            )
        cameras.append({
            "name": f"custom_{i}",
            "pos": tuple(values[:3]),
            "target": tuple(values[3:]),
        })
    return cameras


def look_at_to_quat(eye: tuple, target: tuple) -> tuple:
    """Compute quaternion (w, x, y, z) for a camera at `eye` looking at `target`.

    Isaac Sim cameras look along -Z in their local frame, with Y up.
    """
    ex, ey, ez = eye
    tx, ty, tz = target
    # Forward = normalize(target - eye)
    fx, fy, fz = tx - ex, ty - ey, tz - ez
    flen = math.sqrt(fx * fx + fy * fy + fz * fz)
    if flen < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    fx, fy, fz = fx / flen, fy / flen, fz / flen

    # Camera convention: camera looks along -Z, so we need the rotation
    # that maps -Z to forward direction.
    # Use rotation matrix approach: build basis vectors
    # right = normalize(forward x world_up)
    # up = normalize(right x forward)
    up_world = (0.0, 0.0, 1.0)
    rx = fy * up_world[2] - fz * up_world[1]
    ry = fz * up_world[0] - fx * up_world[2]
    rz = fx * up_world[1] - fy * up_world[0]
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rlen < 1e-8:
        # Looking straight up/down, use Y as world up fallback
        up_world = (0.0, 1.0, 0.0)
        rx = fy * up_world[2] - fz * up_world[1]
        ry = fz * up_world[0] - fx * up_world[2]
        rz = fx * up_world[1] - fy * up_world[0]
        rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rlen, ry / rlen, rz / rlen

    # up = right x forward (not forward x right, since camera -Z is forward)
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx

    # Rotation matrix (columns = right, up, -forward in camera frame)
    # Camera frame: X=right, Y=up, Z=-forward (camera looks along -Z)
    # R = [rx ux -fx]
    #     [ry uy -fy]
    #     [rz uz -fz]
    m00, m01, m02 = rx, ux, -fx
    m10, m11, m12 = ry, uy, -fy
    m20, m21, m22 = rz, uz, -fz

    # Convert rotation matrix to quaternion
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


# =============================================================================
# Scene Configuration
# =============================================================================

def build_scene_cfg(custom_cameras: list[dict]) -> type:
    """Dynamically build scene config class with optional custom cameras."""

    @configclass
    class ReplaySceneCfg(InteractiveSceneCfg):
        """Scene with Koch arm + object + cameras for replay."""

        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )

        koch = KOCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Koch")

        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.15, OBJ_Z)),
        )

        # Top camera
        camera_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/CameraTop",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.15, 0.6),
                rot=(0.0, 0.0, 1.0, 0.0),
                convention="world",
            ),
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            data_types=["rgb"],
            update_period=0,
        )

        # Wrist camera
        camera_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Koch/link4_1/CameraWrist",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.02),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            data_types=["rgb"],
            update_period=0,
        )

    # Add custom cameras as class attributes
    for cam in custom_cameras:
        quat = look_at_to_quat(cam["pos"], cam["target"])
        cam_cfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera_" + cam["name"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=cam["pos"],
                rot=quat,
                convention="world",
            ),
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            data_types=["rgb"],
            update_period=0,
        )
        setattr(ReplaySceneCfg, f"camera_{cam['name']}", cam_cfg)

    return ReplaySceneCfg


# =============================================================================
# Episode Loading
# =============================================================================

def load_episode(dataset_dir: str, episode_idx: int) -> dict:
    """Load an episode from HDF5 file.

    Returns dict with keys: action, state, gripper, object_pos, and metadata.
    """
    filepath = os.path.join(dataset_dir, f"episode_{episode_idx:06d}.hdf5")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Episode file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        data = {
            "action": f["action"][:],                      # (T, 6)
            "state": f["observation/state"][:],             # (T, 6)
            "gripper": f["observation/gripper"][:],         # (T, 1)
            "object_pos": f["object_pos"][:],               # (T, 3)
            "success": bool(f.attrs.get("success", False)),
            "num_steps": int(f.attrs.get("num_steps", f["action"].shape[0])),
            "fps": int(f.attrs.get("fps", 30)),
        }
    print(f"  Loaded {filepath}: {data['num_steps']} steps, success={data['success']}")
    return data


def list_episodes(dataset_dir: str) -> list[int]:
    """List all available episode indices in the dataset directory."""
    pattern = os.path.join(dataset_dir, "episode_*.hdf5")
    files = sorted(glob.glob(pattern))
    indices = []
    for f in files:
        basename = os.path.basename(f)
        idx = int(basename.replace("episode_", "").replace(".hdf5", ""))
        indices.append(idx)
    return indices


# =============================================================================
# Replay Logic
# =============================================================================

def replay_episode(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    episode_data: dict,
    speed: float = 1.0,
):
    """Replay a single episode by driving the robot through recorded actions.

    Args:
        sim: Simulation context.
        scene: Interactive scene.
        episode_data: Dict from load_episode().
        speed: Playback speed multiplier (>1 = faster).
    """
    robot: Articulation = scene["koch"]
    cube = scene["cube"]
    sim_dt = sim.get_physics_dt()

    actions = episode_data["action"]       # (T, 6)
    object_positions = episode_data["object_pos"]  # (T, 3)
    num_steps = episode_data["num_steps"]

    # --- Reset robot to home ---
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.reset()

    # --- Place object at initial recorded position ---
    if len(object_positions) > 0:
        init_obj_pos = object_positions[0]
        pose = torch.tensor(
            [[init_obj_pos[0], init_obj_pos[1], init_obj_pos[2], 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=cube.device,
        )
        pose[:, :3] += scene.env_origins
        cube.write_root_pose_to_sim(pose)
        zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=cube.device)
        cube.write_root_velocity_to_sim(zero_vel)

    sim.step()
    scene.update(sim_dt)

    # --- Compute step skip for speed ---
    step_skip = max(1, int(speed))

    # --- Drive through recorded actions ---
    for t in range(0, num_steps, step_skip):
        if not simulation_app.is_running():
            return

        action = torch.tensor(actions[t], dtype=torch.float32, device=robot.device).unsqueeze(0)
        robot.set_joint_position_target(action)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if t % 30 == 0:
            print(f"    Step {t}/{num_steps}")

    print(f"    Replay complete ({num_steps} steps)")


# =============================================================================
# Main
# =============================================================================

def main():
    
    # print parsed arguments
    print(f"[INFO]: Starting replay with arguments:")
    print(f"  Dataset directory: {args_cli.dataset}")
    if args_cli.all:
        print(f"  Replay mode: ALL episodes")
    else:
        print(f"  Replay mode: Single episode {args_cli.episode}")
    print(f"  Playback speed: {args_cli.speed}x")
    
    
    # Parse custom cameras
    custom_cameras = parse_camera_args(args_cli.camera)
    if custom_cameras:
        print(f"[INFO]: Adding {len(custom_cameras)} custom camera(s)")
        for cam in custom_cameras:
            print(f"  - {cam['name']}: pos={cam['pos']} → target={cam['target']}")

    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.5, 0.5, 0.5], [0.0, 0.0, 0.15])

    # Build scene with custom cameras
    SceneCfg = build_scene_cfg(custom_cameras)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Scene setup complete.")

    dataset_dir = args_cli.dataset

    if args_cli.all:
        # Replay all episodes
        episodes = list_episodes(dataset_dir)
        if not episodes:
            print(f"[ERROR]: No episodes found in {dataset_dir}")
            return
        print(f"[INFO]: Replaying {len(episodes)} episodes from {dataset_dir}")
        for ep_idx in episodes:
            if not simulation_app.is_running():
                break
            print(f"\n[Episode {ep_idx}]")
            episode_data = load_episode(dataset_dir, ep_idx)
            replay_episode(sim, scene, episode_data, speed=args_cli.speed)
    else:
        # Replay single episode
        print(f"[INFO]: Replaying episode {args_cli.episode} from {dataset_dir}")
        episode_data = load_episode(dataset_dir, args_cli.episode)
        replay_episode(sim, scene, episode_data, speed=args_cli.speed)

    print("[INFO]: Replay finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()
