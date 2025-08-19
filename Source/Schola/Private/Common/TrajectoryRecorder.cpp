// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/TrajectoryRecorder.h"

void UTrajectoryRecorder::TickComponent(float DeltaTime, enum ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (!InteractionManager)
	{
		UE_LOG(LogSchola, Warning, TEXT("No InteractionManager found on %s"), *GetOwner()->GetName());
		return;
	}

	if (bRecordData)
	{
		// Check for record termination
		if (TrajectoryEpisode.Steps.Num() >= StepsToRecord)
		{

			bRecordData = false;

			TMap<FString, FDiscreteSpace> DiscreteObservationSpaces;
			TMap<FString, FBinarySpace>	  BinaryObservationSpaces;
			TMap<FString, FBoxSpace>	  BoxObservationSpaces;
			TMap<FString, FDiscreteSpace> DiscreteActionSpaces;
			TMap<FString, FBinarySpace>	  BinaryActionSpaces;
			TMap<FString, FBoxSpace>	  BoxActionSpaces;

			FString JsonString;

			for (FString& Label : InteractionManager->InteractionDefn.ObsSpaceDefn.Labels)
			{
				TSpace& Space = InteractionManager->InteractionDefn.ObsSpaceDefn[Label];
				if (Space.IsType<FDiscreteSpace>())
				{
					DiscreteObservationSpaces.Add(Label, Space.Get<FDiscreteSpace>());
				}
				else if (Space.IsType<FBinarySpace>())
				{
					BinaryObservationSpaces.Add(Label, Space.Get<FBinarySpace>());
				}
				else if (Space.IsType<FBoxSpace>())
				{
					BoxObservationSpaces.Add(Label, Space.Get<FBoxSpace>());
				}
			}

			for (FString& Label : InteractionManager->InteractionDefn.ActionSpaceDefn.Labels)
			{
				TSpace& Space = InteractionManager->InteractionDefn.ActionSpaceDefn[Label];
				if (Space.IsType<FDiscreteSpace>())
				{
					DiscreteActionSpaces.Add(Label, Space.Get<FDiscreteSpace>());
				}
				else if (Space.IsType<FBinarySpace>())
				{
					BinaryActionSpaces.Add(Label, Space.Get<FBinarySpace>());
				}
				else if (Space.IsType<FBoxSpace>())
				{
					BoxActionSpaces.Add(Label, Space.Get<FBoxSpace>());
				}
			}

			FTrajectoryWithDefinition TrajectoryWithDefn = FTrajectoryWithDefinition(
				DiscreteObservationSpaces,
				BinaryObservationSpaces,
				BoxObservationSpaces,
				DiscreteActionSpaces,
				BinaryActionSpaces,
				BoxActionSpaces,
				TrajectoryEpisode);
			FJsonObjectConverter::UStructToJsonObjectString(TrajectoryWithDefn, JsonString);
			UE_LOG(LogSchola, Log, TEXT("Writing agent trajectory to file, this may take a while..."));

			FString LogFileName = FPaths::Combine(OutputDirectory.Path, OutputDataFileName);
			WriteDataToFile(JsonString, LogFileName);
		}
		else
		{
			FTrajectoryStep Step = this->InteractionManager->TrajectoryStep;
			Step.Reward = ComputeReward();
			this->TrajectoryEpisode.Steps.Add(Step);
			CurrStep = IsEpisodeComplete() ? 0 : CurrStep + 1;
			this->InteractionManager->TrajectoryStep = FTrajectoryStep(this->CurrStep, {}, {});
		}
	}
}

void UTrajectoryRecorder::BeginPlay()
{
	Super::BeginPlay();

	IInferenceAgent*  Agent = Cast<IInferenceAgent>(GetOwner());
	AAbstractTrainer* Trainer = Cast<AAbstractTrainer>(GetOwner());
	if (Agent)
	{
		InteractionManager = Agent->GetInteractionManager();
	}
	else if (Trainer)
	{
		InteractionManager = Trainer->InteractionManager;
	}
	else
	{
		UE_LOG(LogSchola, Error, TEXT("No InteractionManager found on %s"), *GetOwner()->GetName());
	}
}

void UTrajectoryRecorder::WriteDataToFile(const FString& DataToSave, const FString& LogFileName)
{
	FFileHelper::SaveStringToFile(DataToSave, *LogFileName, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
}