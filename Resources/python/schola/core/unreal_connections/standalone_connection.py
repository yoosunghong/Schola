# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
A connection that launches an Environment in a new Process.
"""
import logging
import subprocess
import sys
from typing import List, Optional
from schola.core.unreal_connections.base_connection import BaseUnrealConnection


class StandaloneUnrealConnection(BaseUnrealConnection):
    """
    A standalone connection that launches an Unreal Engine instance when started.

    Parameters
    ----------
    url : str
        The URL to connect to
    ue_path : str
        The path to the Unreal Engine executable
    headless_mode : bool
        Whether to run in headless mode
    port : int, optional
        The port to connect to, if None, an open port will be found
    map : str, optional
        The map to load. Defaults to the default map in the Unreal Engine project
    display_logs : bool, default=True
        Whether to display logs
    set_fps : int, optional
        Use a fixed fps while running, if None, no fixed timestep is used
    disable_script : bool, default=False
        Whether to disable the autolaunch script setting in the Unreal Engine Schola Plugin

    Attributes
    ----------
    executable_path : str
        The path to the Unreal Engine executable
    headless_mode : bool
        Whether to run in headless mode
    display_logs : bool
        Whether to display logs
    set_fps : int
        Use a fixed fps while running, if None, no fixed timestep is used
    env_process : subprocess.Popen, optional
        The process running the Unreal Engine. None if the process is not running
    disable_script : bool
        Whether to disable the autolaunch script setting in the Unreal Engine Schola Plugin
    map : str
        The map to load.  Defaults to the default map in the Unreal Engine project
    tcp_socket : socket.socket, optional
        A socket object bound to the open port. None if the port is supplied, and we don't need to open a new port

    """

    def __init__(
        self,
        url: str,
        executable_path: str,
        headless_mode: bool,
        port: Optional[int] = None,
        map: str = None,
        display_logs: bool = True,
        set_fps: Optional[int] = None,
        disable_script: bool = False,
    ):
        if port is None:
            self.tcp_socket, port = self.get_open_port(url)
        else:
            self.tcp_socket = None
            port = port
        super().__init__(url, port)
        self.executable_path = executable_path
        self.headless_mode = headless_mode
        self.display_logs = display_logs
        self.set_fps = set_fps
        self.env_process = None
        self.disable_script = disable_script
        # Note any maps we want to use here need to be added to the build via Project Settings>Packaging>Advanced> List of Maps...
        # or on the command line with the -Map flag for UnrealAutomationTool
        self.map = map

    def make_args(self) -> List[str]:
        """
        Make the arguments supplied to the Unreal Engine Executable.

        Returns
        -------
        List[str]
            The arguments to be supplied to the Unreal Engine Executable
        """
        args = [self.executable_path, "-UNATTENDED"]
        if self.headless_mode:
            args += ["-nullRHI"]
        else:
            args += ["-WINDOWED"]

        if self.map:
            args += [self.map]
        if self.display_logs:
            args += ["-LOG"]
        if not self.set_fps is None:
            args += ["-BENCHMARK"]
            args += ["-FPS=" + str(self.set_fps)]
        args += ["-ScholaPort", str(self.port)]
        if self.disable_script:
            args += ["-ScholaDisableScript"]
        return args

    def start(self) -> None:
        """
        Start the Unreal Engine process.

        Raises
        ------
        Exception
            If the subprocess is already running
        """
        if self.env_process != None:
            raise Exception("Subprocess already running")
        super().start()
        args = self.make_args()
        self.env_process = subprocess.Popen(args)

    def close(self) -> None:
        """
        Close the connection to the Unreal Engine. Kills the Unreal Engine process if it is running.
        """
        super().close()
        PROCESS_KILL_TIMEOUT=1

        if self.tcp_socket != None:
            self.tcp_socket.close()

        if self.env_process != None:
            logging.debug("Killing subprocess")
            self.env_process.kill()
            try:
                self.env_process.wait(timeout=PROCESS_KILL_TIMEOUT) 
            except subprocess.TimeoutExpired:
                if self.env_process.poll() is None:
                    logging.warning("Subprocess.kill() failed, forcibly killing subprocess")
                    if sys.platform.startswith("win"):
                        subprocess.run(f"TASKKILL /F /PID {self.env_process.pid} /T", check=False)
                    else:
                        subprocess.run(["kill", "-9", str(self.env_process.pid)], check=False)

    @property
    def is_active(self) -> bool:
        # channel is still active and the unreal engine is still running
        return (
            super().is_active
            and (not self.env_process is None)
            and (self.env_process.poll() == None)
        )
