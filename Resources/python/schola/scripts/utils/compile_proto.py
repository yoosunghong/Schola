# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
This script is used to compile .proto files into python/C++ Source Files for Schola.
"""
from pathlib import Path
import subprocess
import os
import argparse
from os.path import isfile, join
import re
from typing import List


def get_files(folder):
    return [
        file_name for file_name in os.listdir(folder) if isfile(join(folder, file_name))
    ]


def get_proto_files(folder):
    return [
        file_name for file_name in get_files(folder) if file_name.endswith(".proto")
    ]


def get_generated_cpp_file_types(folder):
    files = get_files(folder)
    output = {
        "proto-header": [],
        "grpc-header": [],
        "proto-c": [],
        "grpc-c": [],
    }

    for file_name in files:

        if file_name.endswith(".grpc.pb.cc"):
            output["grpc-c"].append(file_name)
        elif file_name.endswith(".pb.cc"):
            output["proto-c"].append(file_name)
        elif file_name.endswith(".grpc.pb.h"):
            output["grpc-header"].append(file_name)
        elif file_name.endswith(".pb.h"):
            output["proto-header"].append(file_name)
    return output


def get_generated_python_file_types(folder):
    files = get_files(folder)
    output = {
        "proto": [],
        "grpc": [],
    }

    for file_name in files:
        if file_name.endswith("_pb2_grpc.py"):
            output["grpc"].append(file_name)
        elif file_name.endswith("_pb2.py"):
            output["proto"].append(file_name)

    return output


def fix_imports(folder):
    files = get_files(folder)
    import_pattern = "^import (.*) as (.*)"
    for file in files:
        with open(join(folder, file), "r+") as f:
            file_contents = f.readlines()
            for i, line in enumerate(file_contents):
                result = re.search(import_pattern, line)
                if result:
                    file_contents[i] = (
                        f"import schola.generated.{result.group(1)} as {result.group(2)}\n"
                    )
            f.seek(0)
            f.writelines(file_contents)
            f.truncate()


def disable_warnings(folder, file_paths, warnings):
    for file_path in file_paths:
        with open(join(folder, file_path), "r+") as f:
            file_contents = f.readlines()
            # files have two lines of headers to start
            try:
                headers_start_index = next(
                    (
                        line_num
                        for line_num, line in enumerate(file_contents)
                        if line.startswith("#include ")
                    )
                )
            except StopIteration:
                print(file_path)
                headers_start_index = 3
            for warning in warnings:
                file_contents.insert(
                    headers_start_index, f"#pragma warning (disable : {warning})\n"
                )
            f.seek(0)
            f.writelines(file_contents)
            f.truncate()


def make_proto_files(
    protoc_path, proto_folder, python_folder, cpp_folder, add_type_stubs=False
):
    args = []
    args += [f"-I={proto_folder}"]
    args += [f"--python_out={python_folder}"]
    args += [f"--cpp_out={cpp_folder}"]

    args += [f"--pyi_out={python_folder}"] if add_type_stubs else []

    for file in get_proto_files(proto_folder):
        subprocess.run([protoc_path] + args + [join(proto_folder, file)], check=True)


def make_grpc_files(protoc_path, proto_folder, plugin_path, target_folder):

    I_arg = f"-I={proto_folder}"
    grpc_out = f"--grpc_out={target_folder}"
    plugin = f"--plugin=protoc-gen-grpc={plugin_path}"

    for file in get_proto_files(proto_folder):
        subprocess.run(
            [protoc_path, I_arg, grpc_out, plugin, join(proto_folder, file)], check=True
        )


def make_parser():
    parser = argparse.ArgumentParser("Compile Protobuf Files for Schola")
    parser.add_argument(
        "--plugin-folder",
        type=Path,
        default=Path("."),
        help="Path to the project folder, can be left blank if running from Schola Plugin directory",
    )
    parser.add_argument(
        "--disable-warnings", nargs="+", type=str, default=["4125", "4800"]
    )
    parser.add_argument("--add-type-stubs", action="store_true")
    return parser


def main(plugin_folder: Path, warnings_to_disable: List[str], add_type_stubs: bool):
    """
    Compile Protobuf files for Schola

    Parameters
    ----------
    project : Path
        Path to the project folder
    warnings_to_disable : List[str]
        List of warnings to disable

    """
    plugin_folder = plugin_folder
    proto_folder = plugin_folder / "Proto"
    tools_path = plugin_folder / "Resources" / "tools"
    protoc_path = tools_path / "protoc.exe"
    python_plugin_path = tools_path / "grpc_python_plugin.exe"
    cpp_plugin_path = tools_path / "grpc_cpp_plugin.exe"
    cpp_code_folder = plugin_folder / "Source" / "Schola" / "Generated"
    python_code_folder = plugin_folder / "Resources" / "python" / "Schola" / "generated"

    short_dep_path = r"Schola\Resources\Build\windows_dependencies.bat"

    # Check if protoc_path exists
    if not protoc_path.exists():
        raise FileNotFoundError(
            f"Protoc Path {protoc_path} does not exist. Please run {short_dep_path} to generate this. Please note that Linux is not supported."
        )

    # Check if plugin paths exist
    if not python_plugin_path.exists():
        raise FileNotFoundError(
            f"Python Plugin Path {python_plugin_path} does not exist. Please run {short_dep_path} to generate this. Please note that Linux is not supported."
        )

    if not cpp_plugin_path.exists():
        raise FileNotFoundError(
            f"C++ Plugin Path {cpp_plugin_path} does not exist. Please run {short_dep_path} to generate this. Please note that Linux is not supported."
        )

    # generate protobuf files defining serialization for the messages
    make_proto_files(
        protoc_path, proto_folder, python_code_folder, cpp_code_folder, add_type_stubs
    )

    # generate source for the various message services
    make_grpc_files(protoc_path, proto_folder, python_plugin_path, python_code_folder)
    make_grpc_files(protoc_path, proto_folder, cpp_plugin_path, cpp_code_folder)

    generated_cpp_files = get_generated_cpp_file_types(cpp_code_folder)
    generated_python_files = get_generated_python_file_types(python_code_folder)

    # need to disable safe to ignore warnings that would otherwise cause Unreal compilation errors
    disable_warnings(
        cpp_code_folder, generated_cpp_files["proto-c"], warnings_to_disable
    )

    # generated code doesn't import correctly so we need to prepend Schola.generated._____
    fix_imports(python_code_folder)


def main_from_cli():
    parser = make_parser()
    args = parser.parse_args()
    main(args.plugin_folder, args.disable_warnings, args.add_type_stubs)


if __name__ == "__main__":
    main_from_cli()
