// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "HAL/FileManager.h"
#include "Components/ActorComponent.h"
#include "Environment/EnvironmentComponents/AbstractEnvironmentUtilityComponent.h"
#include "Common/LogSchola.h"
#include "GameFramework/Actor.h"
#include "Training/StateStructs/TrainerState.h"
#include "StatLoggerComponent.generated.h"

UCLASS(Blueprintable, ClassGroup = Schola, meta = (BlueprintSpawnableComponent))
class SCHOLA_API UStatLoggerComponent : public UBlueprintEnvironmentUtilityComponent
{
	GENERATED_BODY()

public:
	/** A map from agent ID to its reward this episode */
	UPROPERTY()
	TMap<int, float> AgentReward = TMap<int, float>();

	/**
	 * @brief The directory to save the log file to.
	 */
	UPROPERTY(EditAnywhere, meta = (RelativeToGameDir), Category = "Logging")
	FDirectoryPath LogDirectory = FDirectoryPath{};

	/** Can the log file be overwritten */
	UPROPERTY(EditAnywhere, Category = "Logging")
	bool bAllowOverwritting = true;

	/**
	 * @brief Log Text to the logfile
	 * @param TextToSave The text to save to the file
	 * @return True if the log was successful
	 */
	UFUNCTION(BlueprintCallable, Category="Logging")
	bool LogToFile(FString TextToSave = "");

	/**
	 * @brief Log a reward to the log file
	 * @param Reward The reward to log
	 * @return True if the log was successful
	 */
	bool LogReward(float Reward);

	void OnEnvironmentReset() override;

	void OnEnvironmentStep(int AgentID, FTrainerState& State) override;

	void OnAgentRegister(int AgentID) override;

	void OnEnvironmentInit(int Id) override;

private:
	/** flag for if this is the first write to the log file */
	UPROPERTY()
	bool bFirstWrite = true;

	/** The path to the log file. Created from the supplied LogDir */
	UPROPERTY()
	FString LogFilePath;
};
