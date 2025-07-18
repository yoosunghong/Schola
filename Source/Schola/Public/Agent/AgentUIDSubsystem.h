// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include <Kismet/GameplayStatics.h>
#include "AgentUIDSubsystem.generated.h"

/**
* @brief A subsystem that manages assigning unique numbered ids to entities.
*/
UCLASS()
class SCHOLA_API UAgentUIDSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

private:
	UPROPERTY()
	int CurrentId;

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/**
	* @brief Get a new unique Id.
	* @returns The unique integer id.
	*/
	UFUNCTION()
	int GetId();
};
