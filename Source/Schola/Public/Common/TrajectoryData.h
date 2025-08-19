// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "TrajectoryData.generated.h"

USTRUCT(BlueprintType)
struct FTrajectoryData
{
	GENERATED_BODY()

	FTrajectoryData(FString InteractorName = TEXT("interactor_name"), FString Value = TEXT(""))
		: InteractorName(InteractorName), Value(Value) {}

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	FString InteractorName = TEXT("interactor_name");

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	FString Value = TEXT("");
};

USTRUCT(BlueprintType)
struct FTrajectoryStep
{
	GENERATED_BODY()

	FTrajectoryStep(int StepNum = 0, TArray<FTrajectoryData> Observations = {}, TArray<FTrajectoryData> Actions = {})
		: StepNum(StepNum), Observations(Observations), Actions(Actions) {}

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	int StepNum = 0;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TArray<FTrajectoryData> Observations;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TArray<FTrajectoryData> Actions;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	float Reward = 0.0f;
};

USTRUCT(BlueprintType)
struct FTrajectoryEpisode
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TArray<FTrajectoryStep> Steps;
};

USTRUCT(BlueprintType)
struct FTrajectoryWithDefinition
{
	GENERATED_BODY()

	FTrajectoryWithDefinition(TMap<FString, FDiscreteSpace> DiscreteObservationSpaces = TMap<FString, FDiscreteSpace>(),
		TMap<FString, FBinarySpace>							BinaryObservationSpaces = TMap<FString, FBinarySpace>(),
		TMap<FString, FBoxSpace>							BoxObservationSpaces = TMap<FString, FBoxSpace>(),
		TMap<FString, FDiscreteSpace>						DiscreteActionSpaces = TMap<FString, FDiscreteSpace>(),
		TMap<FString, FBinarySpace>							BinaryActionSpaces = TMap<FString, FBinarySpace>(),
		TMap<FString, FBoxSpace>							BoxActionSpaces = TMap<FString, FBoxSpace>(),
		FTrajectoryEpisode									Trajectory = FTrajectoryEpisode())
		: DiscreteObservationSpaces(DiscreteObservationSpaces), BinaryObservationSpaces(BinaryObservationSpaces), BoxObservationSpaces(BoxObservationSpaces), DiscreteActionSpaces(DiscreteActionSpaces), BinaryActionSpaces(BinaryActionSpaces), BoxActionSpaces(BoxActionSpaces), Trajectory(Trajectory) {}

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TMap<FString, FDiscreteSpace> DiscreteObservationSpaces;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TMap<FString, FBinarySpace> BinaryObservationSpaces;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TMap<FString, FBoxSpace> BoxObservationSpaces;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TMap<FString, FDiscreteSpace> DiscreteActionSpaces;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TMap<FString, FBinarySpace> BinaryActionSpaces;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	TMap<FString, FBoxSpace> BoxActionSpaces;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Trajectory Recording")
	FTrajectoryEpisode Trajectory;
};