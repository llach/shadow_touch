import numpy as np
import mujoco as mj
import mujoco.viewer
import pathlib as pa

import gymnasium_robotics
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors import MujocoHandBlockTouchSensorsEnv

env = MujocoHandBlockTouchSensorsEnv(
    # model_path=f"{pa.Path(gymnasium_robotics.__file__).parent.absolute()}/envs/assets/hand/manipulate_16.xml"
)

print(len(env.data.ctrl))

data, model = env.data, env.model
with mj.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    # top / back view
    # viewer.cam.azimuth      = 88
    # viewer.cam.distance     = 0.48
    # viewer.cam.elevation    = -5
    # viewer.cam.lookat       = [0.0168021 , -0.00062594,  0.33406475]

    # tip view
    viewer.cam.azimuth      = 90
    viewer.cam.distance     = 0.75
    viewer.cam.elevation    = -1.53
    viewer.cam.lookat       = [0.02396253, -0.00083829, -0.02099901]

    while viewer.is_running():
        data.ctrl = len(data.ctrl)*[0]

        mj.mj_step(model, data)
        print(viewer.cam)

viewer.close()