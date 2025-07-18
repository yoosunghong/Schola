// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Agent/AgentAction.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Policies/PolicyDecision.h"
#include "Training/StateStructs/TrainingState.h"
#include "Training/DefinitionStructs/TrainingDefinition.h"
#include "./IGymConnector.h"
#include "Environment/AbstractEnvironment.h"
#include "GymConnectors/AutoResetTypeEnum.h"
#include "AbstractGymConnector.generated.h"

/** Enum class representing the status of the connector. */
UENUM(BlueprintType)
enum class EConnectorStatus : uint8
{
	Running	   UMETA(DisplayName = "Running"),    /** The connector is currently running */
	Closed	   UMETA(DisplayName = "Closed"),     /** The connector has been closed */
	Error	   UMETA(DisplayName = "Error"),      /** The connector encountered an error */
	NotStarted UMETA(DisplayName = "NotStarted"), /** The connector has not started yet */
};

//Maybe we switch to the template pattern instead of delegates, but this lets other objects subscribe to evenets.
DECLARE_MULTICAST_DELEGATE(FConnectorStartedDelegate);
DECLARE_MULTICAST_DELEGATE(FConnectorClosedDelegate);
DECLARE_MULTICAST_DELEGATE(FConnectorErrorDelegate);


/**
 * @brief An abstract class for connectors between Unreal Engine and gym environments.
 * @details class provides the basic structure and functionality for connecting Unreal Engine environments
 * 	with external gym environments for training purposes.
 */
UCLASS(Blueprintable, Abstract)
class SCHOLA_API UAbstractGymConnector : public UObject, public IGymConnector
{
	GENERATED_BODY()

public:
	/** The current state update from the environments. */
	FTrainingStateUpdate* CurrentStateUpdate;
	
	/** Delegate for when the connector starts. */
	FConnectorStartedDelegate OnConnectorStarted;

	/** Delegate for when the connector closes. */
	FConnectorClosedDelegate  OnConnectorClosed;

	/** Delegate for when the connector encounters an error. */
	FConnectorErrorDelegate	  OnConnectorError;

	/** The status of the connector */
	UPROPERTY()
	EConnectorStatus Status = EConnectorStatus::Running;

	/** The environments that are currently being trained. */
	UPROPERTY()
	TArray<AAbstractScholaEnvironment*> Environments = TArray<AAbstractScholaEnvironment*>();

	/** The states of the environments that are currently being trained. */
	UPROPERTY()
	FTrainingState TrainingState = FTrainingState();

	/** The shared training definition for the environments that will be trained. */
	UPROPERTY()
	FTrainingDefinition TrainingDefinition = FTrainingDefinition();

	/**
	 * @brief Constructor for the abstract gym connector. Initializes the connector with default values.
	 */
	UAbstractGymConnector();
	
	/**
	 * @brief Initialize this gym connector, setting up services and sending agent definitions.
	 * @param[in] AgentDefinitions - The definitions of the agents that will be trained.
	 * @note This function should be implemented by a derived class.
	 */
	virtual void Init(const FTrainingDefinition& AgentDefinitions);

	/**
	 * @brief Initialize this gym connector, collecting environments and training definitions.
	 * @note Calls the other Init function with the SharedTrainingDefinition.
	 */
	void Init();

	/**
	 * @brief Reset all the environments that have completed.
	 */
	virtual void ResetCompletedEnvironments() override;

	/**
	 * @brief Update the environments with the new state update.
	 * @param[in] StateUpdate The new state update.
	 */
	virtual void UpdateEnvironments(FTrainingStateUpdate& StateUpdate) override;

	/**
	 * @brief Collect all the environment states.
	 */
	virtual void CollectEnvironmentStates();

	/**
	 * @brief Set the status of the connector.
	 * @param[in] NewStatus The new status of the connector.
	 */
	void SetStatus(EConnectorStatus NewStatus);

	/**
	 * @brief Submit environment states to the other end of the connector.
	 * @note This function should be implemented by a derived class.
	 */
	virtual void SubmitEnvironmentStates() PURE_VIRTUAL(UAbstractGymConnector::SubmitEnvironmentStates, return; );

	/**
	 * @brief Resolve the environment state update. Useful for connections that operate off of futures, or otherwise require synchronization.
	 * @return The resolved environment state update.
	 * @note This function should be implemented by a derived class.
	 */
	virtual FTrainingStateUpdate* ResolveEnvironmentStateUpdate() PURE_VIRTUAL(UAbstractGymConnector::ResolveEnvironmentStateUpdate, return nullptr;);

	/**
	 * @brief Submit the initial state of the environment after a reset to the other end of the connector.
	 * @param[in] States The states to submit.
	 * @note This function should be implemented by a derived class.
	 */
	virtual void SubmitPostResetState(const FTrainingState& States) PURE_VIRTUAL(UAbstractGymConnector::SubmitPostResetState, return; );

	/**
	 * @brief Update the connector status based on a state update.
	 * @param[in] StateUpdate The state update to base the new status on.
	 */
	void UpdateConnectorStatus(const FTrainingStateUpdate& StateUpdate);

	/**
	 * @brief Update the connector status based on the last state update.
	 */
	void UpdateConnectorStatus();

	/**
	 * @brief Enable the connector. Useful for multistage setup as it is called after init.
	 * @note This function should be implemented by a derived class.
	 */
	virtual void Enable() PURE_VIRTUAL(UAbstractGymConnector::Enable, return; );

	/**
	 * @brief Check if the connector is ready to start.
	 * @return True if the connector is ready to start.
	 * @note This function should be implemented by a derived class.
	 */
	virtual bool CheckForStart() PURE_VIRTUAL(UAbstractGymConnector::CheckForStart, return true;);

	/**
	 * @brief Get the latest state update.
	 * @return The last state update.
	 */
	virtual FTrainingStateUpdate* GetCurrentStateUpdate() { return this->CurrentStateUpdate; };

	/**
	 * @brief Get if the connector is running.
	 * @return True if the connector is running.
	 */
	bool IsRunning() { return Status == EConnectorStatus::Running; };

	/**
	 * @brief Get if the connector is closed.
	 * @return True if the connector is closed.
	 */
	bool IsNotStarted() { return Status == EConnectorStatus::NotStarted || Status == EConnectorStatus::Closed; };
	
	/**
	 * @brief Collect all the EnvironmentManagers in the simulation.
	 */
	void CollectEnvironments() override;

	/**
	 * @brief Register an environment with the subsystem so that it can be controlled by the subsystem.
	 * @param[in] Env A pointer to the environment to be registered.
	 */
	void RegisterEnvironment(AAbstractScholaEnvironment* Env);

	// Functions for handling auto-reset modes

	virtual EAutoResetType GetAutoResetType() PURE_VIRTUAL(UAbstractGymConnector::GetAutoResetType, return EAutoResetType::Disabled;);

	/**
	 * @brief Automatically resets any environments if they are in a terminal state.
	 */
	void SameStepAutoReset();

	
	virtual void AutoReset() override;

};