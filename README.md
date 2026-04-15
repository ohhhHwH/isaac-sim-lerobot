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

# TODO

ADD Multi-view port - def add_viewport(self, view_name):

