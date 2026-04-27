"""
Piper 机械臂 + Orange USD 键盘控制示例
- UP/DOWN: 切换当前选中的关节
- LEFT/RIGHT: 减小/增大当前关节角度
- Q: 退出
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Piper arm keyboard joint control demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import carb
import omni

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "piper_isaac_sim", "USD", "piper_h_v1.usd")
OBJ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "scenes", "kitchen_with_orange",
                         "assets", "Orange001", "Orange001.usd")

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
class DemoSceneCfg(InteractiveSceneCfg):
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


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.4, 0.4], [0.0, 0.0, 0.1])

    scene = InteractiveScene(DemoSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["piper"]
    num_joints = robot.num_joints
    joint_pos = robot.data.joint_pos.clone()

    kb = KeyboardController()
    kb.start()

    current_joint = 0
    print(f"\n=== Piper Keyboard Demo ===")
    print(f"UP/DOWN: select joint | LEFT/RIGHT: move joint | Q: quit")
    print(f"Current joint: {JOINT_NAMES[current_joint]}\n")

    while simulation_app.is_running():
        if kb.pop("Q"):
            break

        if kb.pop("UP"):
            current_joint = (current_joint - 1) % len(JOINT_NAMES)
            print(f"  -> Joint: {JOINT_NAMES[current_joint]} (idx {current_joint})")

        if kb.pop("DOWN"):
            current_joint = (current_joint + 1) % len(JOINT_NAMES)
            print(f"  -> Joint: {JOINT_NAMES[current_joint]} (idx {current_joint})")

        if kb.held("RIGHT"):
            joint_pos[0, current_joint] += ANGLE_STEP

        if kb.held("LEFT"):
            joint_pos[0, current_joint] -= ANGLE_STEP
        
        if kb.held("R"):
            # 重置场景
            pass

        robot.set_joint_position_target(joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
