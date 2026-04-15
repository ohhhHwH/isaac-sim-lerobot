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
python isaac-sim.py
```

----

实现 关节控制场景下机械臂末端位姿、三维坐标等信息的输出，移动情况下每隔1s更新，若不移动，输出最后的信息后不进行输出

----

添加机械臂移动的demo，实现输入一个三维坐标和末端位姿，机械臂夹爪向三维坐标移动的demo, 实现 c 切换 控制模式

----

实现夹爪上相机跟随视角，在gripper_static_1夹爪模型上添加一个跟随视角，视角朝向夹爪爪子，能够看到夹爪爪子和夹爪前方的空间.保证相机空间坐标系与gripper_static_1空间坐标系的相对位置不改变，切换视角前输出其相对坐标的delta值，并在输出EE Pose时同时输出相机的ee pose


----

<body name="gripper_static_1" pos="0.001353 7e-06 -0.045">
  <camera name="gripper_cam" pos="0 0.08 0" xyaxes="1 0 0 0 0.8 -0.6"/>

  
              <inertial pos="0.00544111 0.000115633 -0.0190522" quat="0.687452 -0.165558 -0.165558 0.687452" mass="0.230086" diaginertia="8.4e-05 7.82036e-05 3.37964e-05"/>
              <joint name="joint5" pos="0 0 0" axis="0 0 -1"/>
              <!-- 夹爪跟随相机：位于关节连接处，朝夹爪方向看 -->
              <camera name="gripper_cam" pos="0 0.08 0" xyaxes="1 0 0 0 0.8 -0.6"/>
              <geom pos="-0.001528 -0.105265 -0.122394" quat="1 0 0 0" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="gripper_static_1"/>
              <geom pos="-0.001528 -0.105265 -0.122394" quat="1 0 0 0" type="mesh" mesh="gripper_static_1"/>
              <body name="gripper_moving_1" pos="-0.0074 -0.00025 -0.01315">
                <inertial pos="0.0010956 0.000310904 -0.0251131" quat="0.997878 0 -0.0651045 0" mass="0.100606" diaginertia="2.72621e-05 1.9e-05 1.17379e-05"/>
                <joint name="joint_gripper" pos="0 0 0" axis="0 -1 0" range="0 1.74533"/>
                <geom pos="0.005872 -0.105015 -0.109244" type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="gripper_moving_1"/>
                <geom pos="0.005872 -0.105015 -0.109244" type="mesh" mesh="gripper_moving_1"/>
              </body>
            </body>



----
完善SimIsaacModel class使其他python文件能够引用这个类进行机械臂的控制


class isaaclab.sensors.Camera[源代码]
基类：SensorBase

The camera sensor for acquiring visual data.

This class wraps over the UsdGeom Camera for providing a consistent API for acquiring visual data. It ensures that the camera follows the ROS convention for the coordinate system.

Summarizing from the replicator extension, the following sensor types are supported:

"rgb": A 3-channel rendered color image.

"rgba": A 4-channel rendered color image with alpha channel.

"distance_to_camera": An image containing the distance to camera optical center.

"distance_to_image_plane": An image containing distances of 3D points from camera plane along camera's z-axis.

"depth": The same as "distance_to_image_plane".

"normals": An image containing the local surface normal vectors at each pixel.

"motion_vectors": An image containing the motion vector data at each pixel.

"semantic_segmentation": The semantic segmentation data.

"instance_segmentation_fast": The instance segmentation data.

"instance_id_segmentation_fast": The instance id segmentation data.


----


实现KeyboardController 和 SimIsaacModel 并完成 main函数，不允许修改KeyboardController和SimIsaacModel中原有的函数说明，优先依照注释中内容来实现代码



TODO:加 UI 窗口 实现切换视角，加 臂上相机，


----

@data_produce_init.py 参考初版设计方案 @isaac_koch.py 参考SimIsaacModel 接口及其定义，完善mian函数，使用isaac_koch中定义的相关接口函数，注意保留原来的注释，生成SimIsaacProduce管理环境中其他物体如方块等物体,仅修改data_produce.py文件，不修改isaac_koch中内容


出现问题，No random objects to remove
