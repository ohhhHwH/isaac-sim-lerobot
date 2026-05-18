# Isaac Sim - Koch Robot Arm Simulation

## Project Structure

```
isaac-sim-lerobot/
├── isaac-sim.py          # Main simulation script
├── urdf/
│   ├── koch.urdf         # Koch arm URDF description (6 joints)
│   └── meshes/           # STL mesh files for each link
│       ├── base_link.stl
│       ├── link1_1.stl
│       ├── link2_1.stl
│       ├── link3_1.stl
│       ├── link4_1.stl
│       ├── gripper_static_1.stl
│       └── gripper_moving_1.stl
├── IsaacLab/             # IsaacLab submodule
└── kochsim/              # IsaacLab extension template
```

## isaac-sim.py Logic

### 1. Startup & Initialization

- Uses `isaaclab.app.AppLauncher` to initialize Isaac Sim (supports `--headless`, `--device` etc.)
- All `omni`/`isaaclab` imports happen **after** `AppLauncher` creates the simulation app

### 2. Robot Configuration (`KOCH_CFG`)

- Loads `urdf/koch.urdf` via `UrdfFileCfg`, which converts URDF to USD at runtime
- `fix_base=True` - base link is fixed to the world (arm doesn't fall)
- All 6 joints use `ImplicitActuatorCfg` with position control (stiffness=400, damping=40)
- Joint list: `joint1` ~ `joint5` (arm) + `joint_gripper` (gripper)

### 3. Scene Setup (`KochSceneCfg`)

- Ground plane + dome light
- Koch arm spawned at origin via `InteractiveScene`
- Camera positioned at `[0.5, 0.5, 0.5]` looking at `[0, 0, 0.15]`

### 4. Keyboard Control (`on_keyboard_event`)

Uses `carb.input` to subscribe to keyboard events:

| Key | Action |
|-----|--------|
| UP | Select previous joint |
| DOWN | Select next joint |
| RIGHT | Increase selected joint angle (+0.05 rad/step) |
| LEFT | Decrease selected joint angle (-0.05 rad/step) |
| R | Reset all joints to default position |

- Supports key repeat (hold to continuously adjust)
- Releases LEFT/RIGHT to stop angle change

### 5. Simulation Loop (`run_simulator`)

Each frame:
1. **Reset check** - on first frame or when `R` is pressed, reset robot to default state
2. **Apply delta** - if LEFT/RIGHT is held, accumulate `joint_delta` to `target_pos[current_joint_idx]`
3. **Set target** - call `robot.set_joint_position_target(target_pos)` to drive joints via PD control
4. **Step** - `scene.write_data_to_sim()` -> `sim.step()` -> `scene.update(sim_dt)`

### 6. Control Flow Diagram

```
main()
  ├── Create SimulationContext
  ├── Create InteractiveScene (ground + light + Koch arm)
  ├── sim.reset()
  └── run_simulator()
        ├── Subscribe keyboard events
        └── Loop:
              ├── Reset if needed
              ├── target_pos[joint] += delta   (from keyboard)
              ├── robot.set_joint_position_target(target_pos)
              ├── scene.write_data_to_sim()
              ├── sim.step()
              └── scene.update(dt)
```

## Usage

```bash
python isaac_kocj.py
```


# Piper 


真机 → 仿真 数据转换结构
1. 关节角度 (Joint Angles)

真机 SDK 原始值 (GetArmJointMsgs)
    │
    │  joint_{i} 单位: 0.001° (千分之一度, millidegrees)
    │
    ▼
录制存储值 (CSV)
    │  joint_{i} / 1000 * 0.0174533
    │  = millidegrees → degrees → radians
    │  单位: rad (弧度)
    │
    ▼
回放到真机
    │  rad / 0.0174533 * 1000
    │  = radians → degrees → millidegrees
    │  传入 piper.JointCtrl(*joints)
    │
    ▼
写入仿真 (Isaac Sim)
    │  CSV中的值已经是 rad，可直接写入
    │  joint_pos_t[0, 0:6] = rads_from_csv
    │  无需额外转换
2. 夹爪 (Gripper)

真机 SDK 原始值 (GetArmGripperMsgs)
    │
    │  grippers_angle 单位: 微米 (μm, 1e-6 m)
    │
    ▼
录制存储值 (CSV)
    │  grippers_angle / 1e6
    │  单位: 米 (m), 表示夹爪开口距离
    │  例: 0.070 = 70mm 全开, 0.0 = 闭合
    │
    ▼
回放到真机
    │  pos[-1] * 1e6 → 微米
    │  传入 piper.GripperCtrl(round(pos[-1] * 1e6), ...)
    │
    ▼
写入仿真 (Isaac Sim)
    │  仿真夹爪是旋转关节，单位 rad
    │  需要映射: 开口距离(m) → 关节弧度
    │  gripper_rad = (csv_value / 0.070) * GRIPPER_MAX_RAD
    │  其中 0.070m = 最大开口, GRIPPER_MAX_RAD = 0.04 rad
3. 总结对照表
数据	SDK 原始	CSV 存储	仿真写入
关节 1-6	0.001° (int)	rad (float)	rad — 直接用
夹爪	μm (int)	m (float)	rad — 需线性映射
4. 当前 set_arm_angles 的输入约定
如果输入来自 CSV 录制文件（弧度），应直接写入仿真，不需要 math.radians() 转换。当前函数接受度数输入，所以从 CSV 读取时需要注意：


从 CSV 读取（弧度） → 直接写入仿真
joint_pos_t[0, :6] = torch.tensor(csv_row[:6])  # 已经是 rad

从用户输入（度数） → 需要转换
joint_pos_t[0, :6] = torch.tensor([math.radians(a) for a in angles_deg])


# TODO

ADD Multi-view port - def add_viewport(self, view_name):


source/leisaac/leisaac/tasks/template/lekiwi_env_cfg.py 中 L53 的 TiledCameraCfg
