// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Misc/CommandLine.h"
#include "Environment/AbstractEnvironment.h"
#include "Common/LogSchola.h"
#include "Communicator/CommunicationManager.h"
#include "Agent/AgentAction.h"
#include "Training/AbstractTrainer.h"
#include "Inference/IInferenceAgent.h"
#include "GymConnectors/AbstractGymConnector.h"
#include <Kismet/GameplayStatics.h>
#include "Subsystem/SubsystemSettings/SubsystemSettings.h"
#include "ScholaManagerSubsystem.generated.h"

/**
 * @brief The core subsystem that coordinates the various parts of the UERL toolkit
 */
UCLASS()
class SCHOLA_API UScholaManagerSubsystem : public UTickableWorldSubsystem
{
	GENERATED_BODY()

private:
	/** Boolean variable tracking whether the subsystem has finished initializing, this is separate from the play button on the unreal editor */
	bool bSubsystemPrepared = false;
	
	/** Boolean Variable tracking whether the subsystem has completed it's initial reset */
	bool bFirstStep = true;

protected:
public:

	/** The gym connector that is currently selected */
	UPROPERTY()
	UAbstractGymConnector* GymConnector;

	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/**
	 * @brief Perform an update the currently running environment. Potentially collecting observations on all agents, and doing actions
	 * @param[in] DeltaTime The time since the last tick.
	 */
	virtual void Tick(float DeltaTime) override;

	virtual ETickableTickType GetTickableTickType() const override;
	virtual TStatId			  GetStatId() const override;

	/**
	 * @brief Prepare the subsystem by doing post BeginPlay setup
	 */
	void PrepareSubsystem();

	virtual bool IsTickable() const;

};
