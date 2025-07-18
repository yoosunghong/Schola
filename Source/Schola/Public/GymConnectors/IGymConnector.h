// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Training/UpdateStructs/TrainingUpdate.h"
#include "GymConnectors/AutoResetTypeEnum.h"
#include "IGymConnector.generated.h"


/**
 * @brief An interface for connectors between unreal and gym.
 */
UINTERFACE(BlueprintType)
class UGymConnector : public UInterface
{
	GENERATED_BODY()
};

/**
 * @brief An interface for connectors between unreal and gym.
 */
class SCHOLA_API IGymConnector
{
	GENERATED_BODY()
public:
	
	/**
	 * @brief Collect the environments that should be trained
	 */
	virtual void CollectEnvironments() PURE_VIRTUAL(IGymConnector::CollectEnvironments, return; );

	/**
	 * @brief Collect the states of the environments that are currently being trained.
	 */
	virtual void CollectEnvironmentStates() PURE_VIRTUAL(IGymConnector::CollectEnvironmentStates, return; );

	/**
	 * @brief Submit the states of the environments to gym.
	 */
	virtual void SubmitEnvironmentStates() PURE_VIRTUAL(IGymConnector::SubmitEnvironmentStates, return; );

	/**
	 * @brief Resolve the state update from gym into a TrainingStateUpdate
	 * @return A ptr to the state update from gym
	 */
	virtual FTrainingStateUpdate* ResolveEnvironmentStateUpdate() PURE_VIRTUAL(IGymConnector::ResolveEnvironmentStateUpdate, return nullptr;);

	/**
	 * @brief Reset the environments that have completed their episodes
	 */
	virtual void ResetCompletedEnvironments() PURE_VIRTUAL(IGymConnector::ResetCompletedEnvironments, return; );

	/**
	 * @brief Update the environments with the actions from the agents
	 * @param[in] StateUpdate The state update from gym
	 */
	virtual void UpdateEnvironments(FTrainingStateUpdate& StateUpdate) PURE_VIRTUAL(IGymConnector::UpdateEnvironments, return; );

	/**
	 * @brief Enable the connector
	 */
	virtual void Enable() PURE_VIRTUAL(IGymConnector::Enable, return; );

	/**
	 * @brief Check if this connector is ready to start training
	 * @return True if the connector is ready to start training
	 */
	virtual bool CheckForStart() PURE_VIRTUAL(IGymConnector::CheckForStart, return true;);

	// Auto Reset helpers
	
	/**
	 * @brief Automatically reset some or all of the sub-environments of the Gym-Connector, to support Vectorized Environments.
	 */
	virtual void AutoReset() PURE_VIRTUAL(IGymConnector::AutoReset, return;);

};
