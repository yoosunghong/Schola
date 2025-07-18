// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Misc/Paths.h"
#include "Subsystem/SubsystemSettings/TrainingSettings.h"
#include "Subsystem/SubsystemSettings/StableBaselines/SB3TrainingSettings.h"
#include "Subsystem/SubsystemSettings/Ray/RLlibTrainingSettings.h"
#include "Subsystem/SubsystemSettings/Custom/CustomTrainingSettings.h"
#include "Interfaces/IPluginManager.h"
#include "Subsystem/SubsystemSettings/LaunchableScript.h"
#include "ScriptSettings.generated.h"

FString WithQuotes(FString Input);
/**
 * @brief The type of a training script.
 */
UENUM(BlueprintType)
enum class EScriptType : uint8
{
	Python UMETA(DisplayName = "Python"),
	Other  UMETA(DisplayName = "Other"),
};

/**
 * @brief The type of a python environment.
 */
UENUM(BlueprintType)
enum class EPythonEnvironmentType : uint8
{
	/** The built-in python interpreter from Unreal Engine */
	BuiltIn UMETA(DisplayName = "Built-in Python Env"),
	/** The default system python */
	SystemPath UMETA(DisplayName = "System PATH Python Env"),
	/** A conda environment with a specified name */
	Conda	UMETA(DisplayName = "Conda Env"),
	/** A virtual environment with a specified path */
	VEnv	UMETA(DisplayName = "Custom Python Path"),
};

/**
 * @brief The type of a python training script to use.
 */
UENUM(BlueprintType)
enum class EPythonScript : uint8
{
	/** use the SB3 launch script included with Schola */
	SB3	   UMETA(DisplayName = "Builtin SB3 Training Script"),
	/** use the RLlib launch script included with Schola */
	RLLIB  UMETA(DisplayName = "Builtin RLlib Training Script"),
	/** use a custom launch script set by the user*/
	Custom UMETA(DisplayName = "Custom Training Script"),
};



/**
 * @brief All settings for autolaunching a script when starting the game.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FScriptSettings
{
	GENERATED_BODY()

public:
	
	/** The type of the script to run (e.g. Python, or Other CLI script) */
	UPROPERTY(Config, EditAnywhere, Category= "Script Settings")
	EScriptType ScriptType = EScriptType::Python;

	/** The Python environment to use when running a python script */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "ScriptType==EScriptType::Python", EditConditionHides), Category = "Script Settings|Python Env Settings")
	EPythonEnvironmentType EnvType = EPythonEnvironmentType::BuiltIn;

	/** The name of the conda environment to run the python script in */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "EnvType==EPythonEnvironmentType::Conda && ScriptType==EScriptType::Python", EditConditionHides), Category = "Script Settings|Python Env Settings")
	FString CondaEnvName;

	/** The path to a python executable if using a non-default executable */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "EnvType==EPythonEnvironmentType::VEnv && ScriptType==EScriptType::Python", EditConditionHides), Category = "Script Settings|Python Env Settings")
	FFilePath CustomPythonPath;

	/** The type of python script to run, options are scripts builtin to Schola, or a user provided script */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "ScriptType==EScriptType::Python", EditConditionHides), Category = "Script Settings|Python Script")
	EPythonScript PythonScriptType = EPythonScript::SB3;

	/** Settings to use when running a custom python script */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "ScriptType==EScriptType::Python && PythonScriptType==EPythonScript::Custom", EditConditionHides), Category = "Script Settings|Python Script")
	FCustomTrainingSettings CustomPythonScriptSettings;

	/** Settings to use when running a custom script */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "ScriptType==EScriptType::Other", EditConditionHides), Category = "Script Settings|Custom Script")
	FCustomTrainingSettings CustomScriptSettings;

	/** Settings to use when running the builtin SB3 script */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "ScriptType==EScriptType::Python && PythonScriptType==EPythonScript::SB3", EditConditionHides, DisplayName = "Builtin SB3 Settings"), Category = "Script Settings|Python Script")
	FSB3TrainingSettings SB3Settings;

	/** Settings to use when running the builtin RLlib script */
	UPROPERTY(Config, EditAnywhere, meta = (EditCondition = "ScriptType==EScriptType::Python && PythonScriptType==EPythonScript::RLlib", EditConditionHides, DisplayName = "Builtin RLlib Settings"), Category = "Script Settings|Python Script")
	FRLlibTrainingSettings RLlibSettings;

	FString GetTrainingArgs(int Port) const;

	/**
	 * @brief Get the path to the script to run
	 * @return The path to the script to run
	 */
	FString GetScriptPath() const;

	FLaunchableScript GetLaunchableScript() const;

	FString GetBuiltInPythonPath() const;

	void EnsureScholaIsInstalled() const;

	virtual ~FScriptSettings();
};
