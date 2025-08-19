# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
A connection to an existing Unreal Editor running in a separate process.
"""

from schola.core.unreal_connections.base_connection import BaseUnrealConnection


class UnrealEditorConnection(BaseUnrealConnection):
    """
    A connection to a running Unreal Editor instance.

    Parameters
    ----------
    url : str
        The URL to connect to
    port : int
        The port on that URL to connect to

    Raises
    ------
    AssertionError
        If the port is not supplied
    """

    def __init__(self, url: str, port: int):
        super().__init__(url, port)
        assert (
            self.port is not None
        ), "Port must be supplied to open a connection to an existing Unreal Process"
