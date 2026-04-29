"""
Piper 机械臂 + Orange USD 键盘控制示例
- UP/DOWN: 切换当前选中的关节
- LEFT/RIGHT: 减小/增大当前关节角度
- Q: 退出
"""
from isaac_piper import KeyboardController, PiperDemoSceneCfg
from isaac_piper import set_arm, set_arm_pos
from isaac_piper import simulation_app, JOINT_NAMES, ANGLE_STEP,USD_PATH, OBJ_PATH
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.4, 0.4], [0.0, 0.0, 0.1])

    scene = InteractiveScene(PiperDemoSceneCfg(num_envs=1, env_spacing=2.0))
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

        set_arm(robot, joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

def remote_control():
    # 远程通过位姿去操控机械臂
    pass

if __name__ == "__main__":
    main()
    simulation_app.close()
