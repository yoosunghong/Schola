# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Utility Functions and classes for working with Unreal Engine's UBT (Unreal Build Tool) system.
"""

import os
from pathlib import Path
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


def get_unreal_platform() -> Literal["Win64", "Linux"]:
    """
    Get the current platform for Unreal Engine.

    Returns
    -------
    str
        The platform string for Unreal Engine (e.g., "Win64", "Linux", "Mac").
    """
    if platform.system() == "Windows":
        return "Win64"
    elif platform.system() == "Linux":
        return "Linux"
    else:
        raise ValueError("Unsupported platform: {}".format(platform.system()))


@dataclass
class UBTCommand:
    """
    Dataclass for constructing Unreal Build Tool (UBT) command line arguments.
    """

    ubt_path: Union[Path, str]  # Path to the Unreal Build Tool executable
    project_file: Union[Path, str]  # Path to the .uproject file
    target_platform: Optional[str] = (
        get_unreal_platform()
    )  # Target platform (e.g., Win64, Linux)
    should_package: bool = True  # Whether to package the build
    should_clean: bool = True  # Whether to clean before building
    should_cook: bool = True  # Whether to cook content
    fast_cook: bool = True  # Use FastCook for quicker iteration
    should_build: bool = True  # Whether to build the project
    no_p4: bool = True  # Disable Perforce integration (-NoP4)
    prereqs: bool = True  # Install prerequisites (-prereqs)
    no_compile: bool = True  # Skip compiling C++ code (-nocompile)
    no_compile_uat: bool = True  # Skip compiling Unreal Automation Tool (-nocompileuat)
    configuration: Literal["Development", "Shipping"] = (
        "Development"  # Build configuration
    )
    no_debug_info: bool = True  # Exclude debug info (-nodebuginfo)
    unattended: bool = True  # Run in unattended mode (-unattended)
    staging_dir: Optional[str] = None  # Directory to stage build output
    force_monolithic: bool = True  # Build as a monolithic executable (-ForceMonolithic)
    maps: Optional[List[str]] = field(
        default_factory=list
    )  # List of maps to cook/package
    stdout: bool = True  # Whether to route build process output to stdout

    @property
    def all_maps(self) -> bool:
        return len(self.maps) == 0

    def build_args(self) -> List[str]:
        """
        Build and return the complete list of UBT command line arguments.

        Returns
        -------
        List[str]
            The complete list of UBT command line arguments.
        """
        args = [self.ubt_path, "BuildCookRun"]

        if self.target_platform:
            args.append(f"-platform={self.target_platform}")

        if self.project_file:
            args.append(f"-project={self.project_file}")

        if self.should_package:
            args.append("-package")

        if self.should_clean:
            args.append("-clean")

        if self.should_cook:
            args.append("-cook")
            if self.fast_cook:
                args.append("-FastCook")

        if self.should_build:
            args.append("-build")

        if self.no_p4:
            args.append("-NoP4")

        if self.prereqs:
            args.append("-prereqs")

        if self.no_compile:
            args.append("-nocompile")

        if self.no_compile_uat:
            args.append("-nocompileuat")

        if self.configuration:
            args.append(f"-configuration={self.configuration}")

        if self.stdout:
            args.append("-stdout")

        if self.no_debug_info:
            args.append("-nodebuginfo")

        if self.unattended:
            args.append("-unattended")

        if self.staging_dir:
            args.append("-stage")
            args.append(f"-stagingdirectory={self.staging_dir}")

        if self.force_monolithic:
            args.append("-ForceMonolithic")

        if self.all_maps:
            args.append("-AllMaps")
        elif self.maps:
            args.append(f"-map={'+'.join(self.maps)}")

        return args


def get_ue_version(project_file: Path) -> str:
    with open(project_file, "r") as f:
        # read the file as JSON
        import json

        project_data = json.load(f)
    # extract the engine version
    ue_version = project_data.get("EngineAssociation", None)
    return ue_version


def get_project_file(project_folder) -> Path:
    for file in os.listdir(project_folder):
        if file.endswith(".uproject"):
            return Path(os.path.join(project_folder, file))
    return None


def get_engine_path_from_sln(sln_file: Path) -> Path:
    with open(sln_file, "r") as f:
        for line in f:
            if '"UnrealBuildTool"' in line:
                # extract the path to the UnrealBuildTool
                parts = line.split(",")
                # Get the path to the engine folder using the UBT entry in the solution file
                return sln_file.parent / Path(parts[1].split("Engine")[0].strip(' "'))


def get_sln_file_from_project(project_folder: Path) -> Optional[Path]:
    for file in os.listdir(project_folder):
        if file.endswith(".sln"):
            return project_folder / file
    return None


def get_ubt_path(project_folder: Path, ue_version: str = "5.5") -> Path:
    # try and load it from a sln file
    sln_file = get_sln_file_from_project(project_folder)
    if sln_file is not None:
        engine_path = get_engine_path_from_sln(sln_file)
        if engine_path is not None:
            return (
                engine_path
                / "Engine"
                / "Build"
                / "BatchFiles"
                / f"RunUAT.{('bat' if platform.system() == 'Windows' else 'sh')}"
            )
    else:
        # if we can't find it in the sln file, try and use a default path
        if platform.system() == "Windows":
            return Path(
                "C:/Program Files/Epic Games/UE_"
                + ue_version
                + "/Engine/Build/BatchFiles/RunUAT.bat"
            )


def build_executable(project_file: str, build_dir: str, ubt_path: str):
    args = UBTCommand(
        ubt_path=ubt_path, project_file=project_file, staging_dir=build_dir
    ).build_args()
    comp_process = subprocess.run(args, capture_output=True)
    return comp_process


def quick_build_unreal_project(
    project_folder: str, build_dir: str, ubt_path: Optional[str] = None
):

    uproject_file = None
    uproject_file = get_project_file(project_folder)
    assert uproject_file is not None, "No .uproject file found in the project folder"
    ue_version = get_ue_version(uproject_file)
    assert (
        ue_version is not None
    ), "Could not determine Unreal Engine version from .uproject file"
    ubt_path = (
        get_ubt_path(project_folder, ue_version) if ubt_path is None else Path(ubt_path)
    )
    assert ubt_path is not None, "Could not find Unreal Build Tool (UBT) path"

    build_executable(
        project_file=str(uproject_file),
        build_dir=build_dir,
        ubt_path=str(ubt_path),
    )

    executable_filename = Path(uproject_file).name.split(".")[0] + (
        ".exe" if platform.system() == "Windows" else ""
    )
    built_path = (
        Path(build_dir)
        / platform.system()
        / "ScholaExamples"
        / "Binaries"
        / get_unreal_platform()
        / executable_filename
    )
    return built_path
