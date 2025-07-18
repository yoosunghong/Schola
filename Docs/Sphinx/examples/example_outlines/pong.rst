Pong
----

The Pong environment features two agents playing a collaborative game of pong. The agents receive a reward every step as long as the ball has not hit the wall behind either agent. The game ends when the ball hits the wall behind either agent.

.. csv-table::
    
    "Num Agents", "2"
    "Observation Space", "DictSpace({'Camera_SCS_SceneColorHDR_RTF_RGBA8_R_W16_H16': make_camera_space(16,16,num_channels=1)})"
    "Action Space", "DictSpace({'Teleport_Y_50,00': DiscreteSpace(3)})"
    "Num Vectorized Copies", "2"


