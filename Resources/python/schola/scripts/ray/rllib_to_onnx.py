# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to convert a Ray model to an ONNX model for use in Unreal Engine.
"""
from ray.rllib.policy.policy import Policy
from schola.ray.utils import export_onnx_from_policy
from argparse import ArgumentParser


def make_parser():
    parser = ArgumentParser(prog="Ray-to-Unreal Onnx Parser")

    parser.add_argument("--policy-checkpoint-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.policy_checkpoint_path:
        policy = Policy.from_checkpoint(args.policy_checkpoint_path)
        export_onnx_from_policy(policy, args.output_path)
