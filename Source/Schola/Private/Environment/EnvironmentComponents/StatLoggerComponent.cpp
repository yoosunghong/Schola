// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Environment/EnvironmentComponents/StatLoggerComponent.h"

bool UStatLoggerComponent::LogToFile(FString TextToSave)
{
	if (!bAllowOverwritting)
	{
		if (FPlatformFileManager::Get().GetPlatformFile().FileExists(*LogFilePath))
		{
			return false;
		}
	}

	// Overwrite the existing file if it's the first write operation.
	if (bFirstWrite)
	{
		bFirstWrite = false;
		return FFileHelper::SaveStringToFile(TextToSave, *LogFilePath);
	}

	return FFileHelper::SaveStringToFile(TextToSave, *LogFilePath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
}

bool UStatLoggerComponent::LogReward(float Reward)
{
	return LogToFile(FString::SanitizeFloat(Reward) + "," + LINE_TERMINATOR);
}

void UStatLoggerComponent::OnEnvironmentReset()
{
	float EpisodeAvgReward = 0.0f;
	for (auto& IdStatPair : AgentReward)
	{
		EpisodeAvgReward += IdStatPair.Value;
		IdStatPair.Value = 0;
	}
	EpisodeAvgReward /= AgentReward.Num();
	if (!LogReward(EpisodeAvgReward))
	{
		UE_LOG(LogSchola, Warning, TEXT("Unable to log reward, please check Log Directory settings"));
	}
}

void UStatLoggerComponent::OnEnvironmentStep(int AgentID, FTrainerState& State)
{
	AgentReward[AgentID] += State.Reward;
}

void UStatLoggerComponent::OnAgentRegister(int AgentID)
{
	AgentReward.Add(AgentID, 0.0f);
}

void UStatLoggerComponent::OnEnvironmentInit(int Id)
{
	EnvId = Id;
	LogFilePath = LogDirectory.Path + "\\" + "Results_Env" + FString::FromInt(EnvId) + "_" + FDateTime::Now().ToString() + ".csv";
}
