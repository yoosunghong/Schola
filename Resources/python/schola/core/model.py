# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved

from typing import Tuple
import torch as th


class ScholaModel(th.nn.Module):
    """
    A PyTorch Module that is compatible with Schola inference.

    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: th.Tensor, state) -> Tuple[th.Tensor, th.Tensor]:
        raise NotImplementedError("forward method must be implemented in subclass")

    def save_as_onnx(self, export_path: str, onnx_oppset: int = 17):
        raise NotImplementedError("save as ONNX method must be implemented in subclass")
