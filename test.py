# your_script.py
from isaacsim import SimulationApp

# omni 导入前
simulation_app = SimulationApp({"headless": False})

# 导入 omni 模块
import omni
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

# ... 你的代码





# 最后关闭
simulation_app.close()

