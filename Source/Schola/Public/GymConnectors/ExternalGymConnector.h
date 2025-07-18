// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "./AbstractGymConnector.h"
#include "Agent/AgentAction.h"
#include "ExternalGymConnector.generated.h"

/**
 * @brief An abstract class for connectors that communicate with gym using futures.
 */
UCLASS(Abstract)
class SCHOLA_API UExternalGymConnector : public UAbstractGymConnector
{
	GENERATED_BODY()

public:

	/** How long should we wait before assuming decision request has failed. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (ClampMin = "0", EditCondition = "bUseTimeout", Units = "s"), Category = "Gym Connector Settings")
	int Timeout = 30;
	
	/** Should we use a timeout for decision requests. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, meta = (InlineEditConditionToggle), Category = "Gym Connector Settings")
	bool bUseTimeout = true;

	
	UExternalGymConnector();

	/**
	 * @brief Request a decision from gym using the current state of the agents from environments
	 * @return A future that will eventually contain decision for all agents in all environments
	 * @note This function is asynchronous and will return immediately
	 * @note This function should be implemented by any derived classes
	 */
	virtual TFuture<FTrainingStateUpdate*> RequestStateUpdate()
		PURE_VIRTUAL(UExternalGymConnector::RequestBatchedDecision, return TFuture<FTrainingStateUpdate*>(););

	/**
	 * @brief Send the state of the environment to gym
	 * @param[in] Value The state of the environment
	 * @note This function should be implemented by any derived classes
	 */
	virtual void SendState(const FTrainingState& Value) PURE_VIRTUAL(UExternalGymConnector::SendState, return; );

	void SubmitEnvironmentStates() override;

	FTrainingStateUpdate* ResolveEnvironmentStateUpdate() override;

};