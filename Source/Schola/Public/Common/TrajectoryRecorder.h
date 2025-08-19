// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/InteractionManager.h"
#include "Common/TrajectoryData.h"
#include "Inference/IInferenceAgent.h"
#include "Training/AbstractTrainer.h"
#include "TrajectoryRecorder.generated.h"

UCLASS(Blueprintable, ClassGroup = "Imitation Learning", meta = (BlueprintSpawnableComponent))
class SCHOLA_API UTrajectoryRecorder : public UActorComponent
{
	GENERATED_BODY()

	UTrajectoryRecorder()
	{
		PrimaryComponentTick.bCanEverTick = true;
		PrimaryComponentTick.bStartWithTickEnabled = true;
	}

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Trajectory Recording")
	bool bRecordData = false;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, meta = (EditCondition = "bRecordData"), Category = "Trajectory Recording")
	FString OutputDataFileName = TEXT("Trajectory.json");

	UPROPERTY(EditAnywhere, meta = (RelativeToGameDir, EditCondition = "bRecordData"), Category = "Trajectory Recording")
	FDirectoryPath OutputDirectory = FDirectoryPath{};

	UPROPERTY(EditAnywhere, meta = (ClampMin = "0", EditCondition = "bRecordData"), Category = "Trajectory Recording")
	int StepsToRecord = 1000;

	UPROPERTY(VisibleAnywhere, Category = "Trajectory Recording")
	int CurrStep = 0;

	/** The most recently collected observations */
	UPROPERTY()
	FDictPoint Observations;

	/** The trajectory data for this interaction manager */
	UPROPERTY()
	FTrajectoryEpisode TrajectoryEpisode = FTrajectoryEpisode();

	/** A pointer to the interaction manager for the trainer or agent that this component is attached to for recording observations and actions */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UInteractionManager* InteractionManager;

	void TickComponent(float DeltaTime, enum ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	void BeginPlay() override;

	void WriteDataToFile(const FString& DataToSave, const FString& LogFileName);

	UFUNCTION(BlueprintImplementableEvent, Category = "Trajectory Recording")
	float ComputeReward();

	UFUNCTION(BlueprintImplementableEvent, Category = "Trajectory Recording")
	bool IsEpisodeComplete();
};