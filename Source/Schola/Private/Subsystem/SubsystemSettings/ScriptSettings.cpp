// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Subsystem/SubsystemSettings/ScriptSettings.h"

inline FString WithQuotes(FString Input)
{
	return FString("\"") + Input + FString("\"");
}

FString FScriptSettings::GetTrainingArgs(int Port) const
{
	FScriptArgBuilder ArgBuilder = FScriptArgBuilder();
	switch (ScriptType)
	{
		case (EScriptType::Python):
			switch (PythonScriptType)
			{
				case (EPythonScript::SB3):
					this->SB3Settings.GenerateTrainingArgs(Port, ArgBuilder);
					break;
				case (EPythonScript::RLLIB):
					this->RLlibSettings.GenerateTrainingArgs(Port, ArgBuilder);
					break;
				default:
					this->CustomPythonScriptSettings.GenerateTrainingArgs(Port, ArgBuilder);
					break;
			}
			break;

		default:
			this->CustomScriptSettings.GenerateTrainingArgs(Port, ArgBuilder);
			break;
	}
	return ArgBuilder.Build();
}

FString FScriptSettings::GetScriptPath() const
{
	switch (ScriptType)
	{
		case (EScriptType::Python):
			switch (PythonScriptType)
			{
				case (EPythonScript::SB3):
					return FString("-m schola.scripts.sb3.launch");

				case (EPythonScript::RLLIB):
					return FString("-m schola.scripts.ray.launch");

				default:
					return WithQuotes(CustomPythonScriptSettings.LaunchScript.FilePath);
			}

		default:
			return WithQuotes(CustomScriptSettings.LaunchScript.FilePath);
	}
}

FLaunchableScript FScriptSettings::GetLaunchableScript() const
{	
	FString ScriptCommand = this->GetScriptPath();
	switch (ScriptType)
	{
		case (EScriptType::Python):
			switch (EnvType)
			{
				case (EPythonEnvironmentType::Conda):
					return FLaunchableScript(FString("conda"), FString("run --live-stream -n ") + this->CondaEnvName + FString(" python ") + ScriptCommand);
				case (EPythonEnvironmentType::VEnv):
					//Convert to absolute file path
					if (this->CustomPythonPath.FilePath.StartsWith(TEXT("..")))
					{
						return FLaunchableScript(FPaths::ConvertRelativePathToFull(this->CustomPythonPath.FilePath), ScriptCommand);
					}
					else
					{
						return FLaunchableScript(this->CustomPythonPath.FilePath, ScriptCommand);
					}

				case (EPythonEnvironmentType::SystemPath):
					return FLaunchableScript(FString("python"), ScriptCommand);

				default:
					EnsureScholaIsInstalled();
					return FLaunchableScript(GetBuiltInPythonPath(), ScriptCommand);
			}

		default:
			return FLaunchableScript(ScriptCommand);
	}
}

FString FScriptSettings::GetBuiltInPythonPath() const
{
#if PLATFORM_WINDOWS
	return FPaths::Combine(FPaths::EngineDir(), TEXT("Binaries/ThirdParty/Python3/Win64/python.exe"));
#elif PLATFORM_MAC
	return FPaths::Combine(FPaths::EngineDir(), TEXT("Binaries/ThirdParty/Python3/Mac/bin/python3"));
#elif PLATFORM_LINUX
	return FPaths::Combine(FPaths::EngineDir(), TEXT("Binaries/ThirdParty/Python3/Linux/bin/python3"));
#else
	UE_LOG(LogSchola, Warning, TEXT("Unsupported platform, defaulting to Linux Python path"));
	return FPaths::Combine(FPaths::EngineDir(), TEXT("Binaries/ThirdParty/Python3/Linux/bin/python3"));
#endif
}

void FScriptSettings::EnsureScholaIsInstalled() const
{
	FString PythonPath = GetBuiltInPythonPath();
	int32 ReturnCode;
	FString StdOut;
	FString StdErr;
	FPlatformProcess::ExecProcess(*PythonPath, TEXT("-m pip show schola"), &ReturnCode, &StdOut, &StdErr);

	// if not installed, install it
	if (ReturnCode != 0)
	{
		FString Command = FString::Printf(TEXT("-m pip install %s[all]"), *(*IPluginManager::Get().FindPlugin(TEXT("Schola"))->GetBaseDir() + FString("/Resources/python")));
		FPlatformProcess::ExecProcess(*PythonPath, *Command, &ReturnCode, &StdOut, &StdErr);
		if (ReturnCode != 0)
		{
			UE_LOG(LogSchola, Error, TEXT("Failed to install Schola python package: %s \n%s"), *StdOut, *StdErr);
		} else{
			UE_LOG(LogSchola, Log, TEXT("Installed Schola python package"));
		}	
	}
}

FScriptSettings::~FScriptSettings()
{
}