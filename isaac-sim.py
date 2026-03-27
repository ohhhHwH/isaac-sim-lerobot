"""Koch robot arm simulation with keyboard joint control.

Usage:
    ./IsaacLab/isaaclab.sh -p isaac-sim.py

Controls:
    UP/DOWN   - Select joint
    LEFT/RIGHT - Decrease/Increase joint angle
    R         - Reset all joints to zero
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Koch arm keyboard control")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import carb
import omni

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

# --- Koch arm config ---
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "koch.urdf")

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

# --- Joint names for display ---
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint_gripper"]

# --- Keyboard state ---
current_joint_idx = 0
joint_delta = 0.0  # accumulated delta per step
ANGLE_STEP = 0.05  # radians per key press (~2.9 degrees)
reset_flag = False

# --- Camera presets: (name, eye, target) ---
CAMERA_PRESETS = [
    ("front",   (0.5, 0.0, 0.3), (0.0, 0.0, 0.15)),
    ("side",    (0.0, 0.5, 0.3), (0.0, 0.0, 0.15)),
    ("top",     (0.0, 0.0, 0.8), (0.0, 0.0, 0.0)),
    ("default", (0.5, 0.5, 0.5), (0.0, 0.0, 0.15)),
    ("follow",  None, None),  # tracks link4_1 from above
]
current_camera_idx = 3  # start with "default"
camera_changed = False


def on_keyboard_event(event, *args, **kwargs):
    """Keyboard callback for joint selection and angle control."""
    global current_joint_idx, joint_delta, reset_flag, current_camera_idx, camera_changed

    if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
        if event.input.name == "UP":
            current_joint_idx = (current_joint_idx - 1) % len(JOINT_NAMES)
            print(f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})")
        elif event.input.name == "DOWN":
            current_joint_idx = (current_joint_idx + 1) % len(JOINT_NAMES)
            print(f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})")
        elif event.input.name == "RIGHT":
            joint_delta = ANGLE_STEP
        elif event.input.name == "LEFT":
            joint_delta = -ANGLE_STEP
        elif event.input.name == "R":
            reset_flag = True
        elif event.input.name == "C":
            current_camera_idx = (current_camera_idx + 1) % len(CAMERA_PRESETS)
            camera_changed = True
            print(f"[Camera] Switched to: {CAMERA_PRESETS[current_camera_idx][0]}")

    if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        if event.input.name in ("LEFT", "RIGHT"):
            joint_delta = 0.0

    return True


# --- Scene config ---
@configclass
class KochSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    koch = KOCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Koch")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    global joint_delta, reset_flag, camera_changed

    robot: Articulation = scene["koch"]
    sim_dt = sim.get_physics_dt()

    # Resolve link4 body index for follow camera
    link4_body_ids, _ = robot.find_bodies("link4_1")
    link4_body_id = link4_body_ids[0]
    FOLLOW_OFFSET = (0.0, 0.0, 0.4)  # above link4

    # Initialize target positions
    target_pos = robot.data.default_joint_pos.clone()

    # Subscribe to keyboard
    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("\n" + "=" * 50)
    print("Koch Arm Keyboard Control")
    print("=" * 50)
    print("  UP/DOWN    : Select joint")
    print("  LEFT/RIGHT : Decrease/Increase angle")
    print("  R          : Reset all joints")
    print("  C          : Cycle camera view")
    print("=" * 50)
    print(f"  Camera views: {', '.join(p[0] for p in CAMERA_PRESETS)}")
    print(f"[Joint] Selected: {JOINT_NAMES[current_joint_idx]} (index {current_joint_idx})")
    print(f"[Camera] Current: {CAMERA_PRESETS[current_camera_idx][0]}\n")

    count = 0
    while simulation_app.is_running():
        # Reset
        if count == 0 or reset_flag:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            target_pos = robot.data.default_joint_pos.clone()
            if reset_flag:
                print("[INFO]: Reset all joints to default position")
                reset_flag = False

        # Apply keyboard delta to the selected joint
        if joint_delta != 0.0:
            target_pos[0, current_joint_idx] += joint_delta

        # Set position target
        robot.set_joint_position_target(target_pos)

        # Update camera view
        cam_name, cam_eye, cam_target = CAMERA_PRESETS[current_camera_idx]
        if cam_name == "follow":
            link4_pos = robot.data.body_pos_w[0, link4_body_id].tolist()
            eye = [link4_pos[0] + FOLLOW_OFFSET[0], link4_pos[1] + FOLLOW_OFFSET[1], link4_pos[2] + FOLLOW_OFFSET[2]]
            sim.set_camera_view(eye, link4_pos)
        elif camera_changed:
            sim.set_camera_view(list(cam_eye), list(cam_target))
            camera_changed = False

        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

    input_iface.unsubscribe_to_keyboard_events(keyboard, sub)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.5, 0.5, 0.5], [0.0, 0.0, 0.15])

    scene_cfg = KochSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
