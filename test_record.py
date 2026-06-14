"""
conda run -n env_isaaclab python test_record.py > test_record.log 2>&1
kill -9 $(pgrep -f "test_record.py")

使用 conda 环境 env_isaaclab
使用 isaac_piper.py 中的类与接口函数
添加两个视频流的获取，视频流放置于 datasets/piper_demo/ 下，命名为 front.mp4 和 side.mp4
        顶部视角-"{ENV_REGEX_NS}/Piper/camera_link/gripper_cam"
        腕部视角-"{ENV_REGEX_NS}/CameraTop"
一段数据：机械臂抓取过程中的关节角度数据轨迹放置于 datasets/piper_demo/ 下，命名为 trajectory.json
    关节角度移动轨迹如下
    1.关节角度: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.99) 末端位姿: [[0.06, 0.0, 0.21], [0.0, 85.0, -0.0]]
    2.关节角度: ([45.01, 93.744, -57.566, 0.0, 58.702, 74.98], 0.0) 末端位置: [[0.2, 0.2, 0.2], [-150.03, 0.09, -179.91]]
    3.关节角度: ([0.0, 3.243, 0.0, 0.0, 0.0, 0.0], 0.99) 末端位置: [[0.06, 0.0, 0.21], [0.0, 88.24, -0.0]]
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# AppLauncher — must execute before remaining isaaclab imports (sets up Omniverse)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Now safe to import isaaclab / project internals
# ---------------------------------------------------------------------------
import numpy as np
import cv2


from isaac_piper import (
    PiperSim,
    PiperDemoSceneCfg,
    CAM_WIDTH,
    CAM_HEIGHT,
    simulation_app,
)

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("datasets/piper_demo")

# Camera → video file mapping (user-specified naming convention)
#   顶部视角 → gripper_cam (mounted on camera_link, wrist-mounted) → front.mp4
#   腕部视角 → top         (static overhead CameraTop)             → side.mp4
CAMERA_VIDEO_MAP = {
    "front": "gripper_cam",
    "side": "top",
}

# Trajectory waypoints in DEGREES (joint angles) and [0, 1] range (gripper).
# End-poses are metadata stored alongside each waypoint.
TRAJECTORY_WAYPOINTS = [
    {
        "joint_angles_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper_open": 0.99,
        "end_pose": {"pos": [0.06, 0.0, 0.21], "euler_deg": [0.0, 85.0, -0.0]},
    },
    {
        "joint_angles_deg": [45.01, 93.744, -57.566, 0.0, 58.702, 74.98],
        "gripper_open": 0.0,
        "end_pose": {"pos": [0.2, 0.2, 0.2], "euler_deg": [-150.03, 0.09, -179.91]},
    },
    {
        "joint_angles_deg": [0.0, 3.243, 0.0, 0.0, 0.0, 0.0],
        "gripper_open": 0.99,
        "end_pose": {"pos": [0.06, 0.0, 0.21], "euler_deg": [0.0, 88.24, -0.0]},
    },
]

# Number of interpolation steps between consecutive waypoints
INTERP_STEPS = 100
# Video output FPS
VIDEO_FPS = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def deg_to_rad(deg: float) -> float:
    return float(deg) * np.pi / 180.0


def get_camera_rgb(scene: InteractiveScene, camera_name: str) -> np.ndarray:
    """Grab the latest RGB frame from a named camera in the scene."""
    camera = scene[camera_name]
    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    return np.asarray(rgb, dtype=np.uint8)


def main() -> None:
    # -- prepare output directory --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- video writers --
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_size = (CAM_WIDTH, CAM_HEIGHT)

    front_writer = cv2.VideoWriter(
        str(OUTPUT_DIR / "front.mp4"), fourcc, VIDEO_FPS, frame_size
    )
    side_writer = cv2.VideoWriter(
        str(OUTPUT_DIR / "side.mp4"), fourcc, VIDEO_FPS, frame_size
    )

    # -- simulation setup (mirrors isaac_piper.py patterns) --
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 200.0,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            enable_ccd=True,
            enable_stabilization=True,
            bounce_threshold_velocity=0.2,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.4, 0.4], [0.0, 0.0, 0.1])

    scene_cfg = PiperDemoSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    piper = PiperSim(scene)

    # -- convert waypoints to simulation units (radians, meters) --
    waypoints_sim = []
    for wp in TRAJECTORY_WAYPOINTS:
        angles_rad = [deg_to_rad(a) for a in wp["joint_angles_deg"]]
        gripper_m = wp["gripper_open"] * piper.GRIPPER_MAX_OPEN_M
        waypoints_sim.append({
            "angles_rad": angles_rad,
            "gripper_open_m": gripper_m,
            "end_pose": wp["end_pose"],
        })

    recorded_trajectory = []  # list of per-step state dictionaries

    def capture_frame() -> None:
        """Write one video frame from each camera and record joint state snapshot."""
        # -- video frames (RGB → BGR for OpenCV) --
        front_rgb = get_camera_rgb(scene, CAMERA_VIDEO_MAP["front"])
        side_rgb = get_camera_rgb(scene, CAMERA_VIDEO_MAP["side"])
        front_writer.write(cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
        side_writer.write(cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR))

        # -- joint state snapshot --
        joint_pos = piper.get_joint_angles()       # (1, 8) tensor
        end_pos, end_quat = piper.get_arm_pos()    # base-frame position & quaternion

        recorded_trajectory.append({
            "step": len(recorded_trajectory),
            "joint_angles_rad": joint_pos[0, :6].cpu().tolist(),
            "gripper_joint_rad": joint_pos[0, 6:8].cpu().tolist(),
            "end_pos": [round(v, 6) for v in end_pos],
            "end_quat": [round(v, 6) for v in end_quat],
        })

    # -- initial settle at first waypoint --
    wp0 = waypoints_sim[0]
    piper.set_arm_angles(angles_rad=wp0["angles_rad"], gripper_open_m=wp0["gripper_open_m"])
    for _ in range(40):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)
    capture_frame()
    print("[0/2] Initial waypoint settled.")

    # -- interpolate between consecutive waypoints --
    for wp_idx in range(len(waypoints_sim) - 1):
        w_start = waypoints_sim[wp_idx]
        w_end = waypoints_sim[wp_idx + 1]

        for t in range(1, INTERP_STEPS + 1):
            alpha = t / INTERP_STEPS

            # linear interpolation in joint space
            angles = [
                w_start["angles_rad"][j]
                + alpha * (w_end["angles_rad"][j] - w_start["angles_rad"][j])
                for j in range(6)
            ]
            gripper = w_start["gripper_open_m"] + alpha * (
                w_end["gripper_open_m"] - w_start["gripper_open_m"]
            )

            piper.set_arm_angles(angles_rad=angles, gripper_open_m=gripper)
            capture_frame()

            if t % 20 == 0:
                print(
                    f"  [{wp_idx}→{wp_idx+1}] {t}/{INTERP_STEPS}  "
                    f"α={alpha:.2f}  gripper={gripper:.4f}m"
                )

        print(f"[{wp_idx}→{wp_idx+1}] segment complete.")

    # -- final settle at last waypoint --
    w_last = waypoints_sim[-1]
    piper.set_arm_angles(
        angles_rad=w_last["angles_rad"], gripper_open_m=w_last["gripper_open_m"]
    )
    for _ in range(60):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)
    capture_frame()
    print("[final] Last waypoint settled.")

    # -- save trajectory JSON --
    traj_path = OUTPUT_DIR / "trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(recorded_trajectory, f, indent=2, ensure_ascii=False)

    # -- cleanup --
    front_writer.release()
    side_writer.release()

    print(f"\nDone!")
    print(f"  Video front : {OUTPUT_DIR / 'front.mp4'}")
    print(f"  Video side  : {OUTPUT_DIR / 'side.mp4'}")
    print(f"  Trajectory  : {traj_path}  ({len(recorded_trajectory)} total steps)")


if __name__ == "__main__":
    main()
    simulation_app.close()
