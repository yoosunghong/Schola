Behaviour Cloning
=================
Behavioural cloning (BC) is a technique that directly learns a policy by using supervised learning on observation-action pairs from expert demonstrations. Schola provides tools for trajectory data collection in Unreal Engine and facilitates the cloning process through integration with Stable Baselines 3 and Ray RLLib. In this guide, we will explain how to collect trajectory data and behaviour clone using Schola.

Creating Trajectory Recorder
----------------------------
In Schola, trajectory data consists of a sequence of ``observations``, ``actions``, ``rewards``, and ``episode termination`` flag. To record this data, you will need to create a blueprint subclass of the :cpp:class:`~UTrajectoryRecorder` class, which will record the observations and actions through the :cpp:class:`~UInteractionManager` that it is attached to, and rewards and episode termination flags throught the overridable the ComputeReward and IsEpisodeComplete functions. Once you have created the blueprint subclass, you can add it to your level in Unreal Engine and configure it to start recording trajectory data.

.. figure:: /_static/guides/bc/trajectory_recorder_bp.png
   :alt: Trajectory recorder creation
   :width: 80%

Configuring the Trajectory Recorder
-----------------------------------
The :cpp:class:`~UTrajectoryRecorder` subclass can be used to collect data during training or inference respectively by being attached to :cpp:class:`~AAbstractTrainer` or :cpp:class:`~IInferenceAgent` classes. In the subclass's blueprint details panel, under ``Trajectory Recording``, you can configure the recorder by toggling the ``Record Trajectory`` checkbox to enable or disable trajectory recording, the ``OutputDataFileName`` and ``OutputDirectory`` to specify the file name and directory where the trajectory data will be saved, and the ``StepsToRecord`` to specify the number of steps to record before stopping.

.. figure:: /_static/guides/bc/details.png
   :alt: Trajectory recorder confguration
   :width: 80%

Creating InferencePlayerController
----------------------------------

To collect human trajectory data, you will also need to create an :cpp:class:`~AInferencePlayerController` subclass. This class implements the :cpp:class:`~IInferenceAgent` interface, so you will be attaching the :cpp:class:`~UTrajectoryRecorder` subclass to it, and have the option to add any `Observers` and `Actuators`. 

.. figure:: /_static/guides/bc/player_controller_bp.png
   :alt: InferencePlayerController creation
   :width: 80%

However, unlike other inference agents, you will need to override ``DoDiscreteAction``, ``DoContinuousAction``, and ``DoBinaryAction`` functions under the `Reinforcement Learning` group to handle mapping the input from the player to actuator actions in Schola. One way to facilitates this mapping is to create variables, change their values on key press events, then read and return the values of these variables in the overridden functions which will pass them to the actuators.

.. figure:: /_static/guides/bc/action_functions.png
   :alt: Overridable functions in InferencePlayerController
   :width: 80%

.. blueprint-file:: bc/event_graph.bp
   :heading: MyInferencePlayerController > EventGraph
   :imagefallback:  /_static/guides/bc/event.png
   :height: 400
   :zoom: -5

.. blueprint-file:: bc/do_discrete_action.bp
   :heading: MyInferencePlayerController > DoDiscreteAction
   :imagefallback:  /_static/guides/bc/override.png
   :height: 400
   :zoom: -5

To start collecting trajectory data, you will need to set the ``AI Controller Class`` in the details panel of the pawn you want to control to your newly created :cpp:class:`~AInferencePlayerController` subclass. 



Running Behaviour Cloning
-------------------------
Once you have collected the trajectory data, you can use it to perform behaviour cloning. To do this, you can follow these steps:

    1. Run the game in Unreal Engine (by clicking the green triangle).
    2. Open a terminal or command prompt, and run the following Python script:

.. tabs::


    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
            
            schola-bc-sb3 -p <PORT_NUMBER> --expert-path <EXPERT_PATH>

    .. group-tab:: Ray
        .. code-block:: bash
            
            schola-bc-rllib -p <PORT_NUMBER> --expert-path <EXPERT_PATH>

The ``<PORT_NUMBER>`` should be replaced with the port number you used to launch the Unreal Engine game, and ``<EXPERT_PATH>`` should be replaced with the path to the trajectory data file you collected earlier. There are also additional options available for the command line interface that allows you to specify the number of training steps, the learning rate, whether to save as onnx, and other parameters. For more information on these options, refer to the documentation for the respective command line interface.