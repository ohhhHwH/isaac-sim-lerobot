"""
Piper 机械臂 + Orange USD 键盘控制示例
- UP/DOWN: 切换当前选中的关节
- LEFT/RIGHT: 减小/增大当前关节角度
- Q: 退出
"""
from isaac_piper import KeyboardController, PiperDemoSceneCfg

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

        robot.set_joint_position_target(joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

def remote_control():
    # 远程通过位姿去操控机械臂
    from isaac_piper import PiperSim
    import socket
    import json
    import torch
    IP = "0.0.0.0"
    PORT = 3456
    CONTROL_MODE = "tor"
    
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0, # 120Hz 更新频率
        device="cuda:0",
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.4, 0.4], [0.0, 0.0, 0.1])

    scene = InteractiveScene(PiperDemoSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    # 创建 PiperSim 实例
    piper = PiperSim(scene)
    
    # 创建 socket 连接
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))
    print(f"Listening on {IP}:{PORT} ...")
    while simulation_app.is_running():
        data, addr = sock.recvfrom(4096)  # 接收数据
        try:
            msg = json.loads(data.decode("utf-8"))
            if CONTROL_MODE == "tor":
                print(f"Received : {msg}")
                pos = msg.get("position") or msg.get("pos")
                rot = msg.get("orientation") or msg.get("quat")
                
                # TODO KOCH -> PIPER 夹爪映射
                gripper_angle = msg["gripper_angle"]
                # "position": pos, "orientation": quat, "gripper_angle": gripper_angle
                pos_t = torch.as_tensor([pos], dtype=torch.float32)
                rot_t = torch.as_tensor([rot], dtype=torch.float32)
                
                piper.set_arm_pos(pos_t, rot_t)
            else:
                pass
        except Exception as e:
            print(f"Error decoding message: {e}")
    

if __name__ == "__main__":
    # main()
    remote_control()
    simulation_app.close()
