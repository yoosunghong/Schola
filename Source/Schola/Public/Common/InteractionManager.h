// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/InteractionDefinition.h"
#include "Agent/AgentComponents/SensorComponent.h"
#include "Observers/AbstractObservers.h"
#include "Components/ActorComponent.h"
#include "Agent/AgentUIDSubsystem.h"
#include "Common/Spaces.h"
#include "Common/Points.h"
#include "Common/LogSchola.h"
#include "Common/TrajectoryData.h"
#include "Actuators/AbstractActuators.h"
#include "Containers/SortedMap.h"
#include "Agent/AgentComponents/ActuatorComponent.h"
#include "JsonObjectConverter.h"
#include "InteractionManager.generated.h"

UCLASS(Blueprintable)
class UInteractionManager : public UObject
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Reinforcement Learning")
	TArray<UAbstractObserver*> Observers;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Reinforcement Learning")
	TArray<UActuator*> Actuators;

	/** The most recently collected observations */
	UPROPERTY()
	FDictPoint Observations;

	/** The input output spaces, and other information for this interaction manager */
	UPROPERTY(VisibleAnywhere, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	FInteractionDefinition InteractionDefn;

	/** The trajectory for the current step. This is used by trajectory recorders who observe the interaction manager */
	UPROPERTY()
	FTrajectoryStep TrajectoryStep = FTrajectoryStep(0, {}, {});

	/**
	 * @brief Setup the observers for this interaction manager
	 * @param[in] InObservers The observers to setup
	 * @param[out] OutObservers A copy of the input observers (used to set the Observers on the InteractionManager)
	 */
	void SetupObservers(const TArray<UAbstractObserver*>& InObservers, TArray<UAbstractObserver*>& OutObservers);

	/**
	 * @brief Collect observations from the observers
	 * @param[in] InObservers The observers to collect observations from
	 * @param[out] OutObservationsMap The collected observations
	 */
	void CollectObservationsFromObservers(const TArray<UAbstractObserver*>& InObservers, FDictPoint& OutObservationsMap);

	/**
	 * @brief Collect observation spaces from a List of Observers
	 * @param[in] InObservers The observers to collect observation spaces from
	 * @param[out] OutSpaceGroups The collected observation spaces
	 */
	void CollectObservationSpaceFromObservers(const TArray<UAbstractObserver*>& InObservers, FDictSpace& OutSpaceGroups);

	/**
	 * @brief Setup the actuators for this interaction manager
	 * @param[in] InActuators The actuators to setup
	 * @param[out] OutActuators A copy of the input actuators (used to set the Actuators on the InteractionManager)
	 */
	void SetupActuators(const TArray<UActuator*>& InActuators, TArray<UActuator*>& OutActuators);

	/**
	 * @brief Send actions to actuators
	 * @param[in] OutActuators The actuators to send actions to
	 * @param[in] Actions The actions to send
	 */
	void SendActionsToActuators(TArray<UActuator*>& OutActuators, const FDictPoint& Actions);

	/**
	 * @brief Collect action spaces from a List of Actuators
	 * @param[in] InActuators The actuators to collect action spaces from
	 * @param[out] OutSpaceGroups The collected action spaces
	 */
	void CollectActionSpaceFromActuators(const TArray<UActuator*>& InActuators, FDictSpace& OutSpaceGroups);

	/**
	 * @brief Initialize the InteractionManager, from a list of observers and actuators
	 * @param[in] InObservers The observers that will be managed, and initialized
	 * @param[in] InActuators The actuators that will be managed, and initialized
	 */
	void Initialize(TArray<UAbstractObserver*>& InObservers, TArray<UActuator*>& InActuators);

	/**
	 * @brief Distribute Actions to the actuators
	 * @param[in] ActionMap The actions to distribute
	 */
	void DistributeActions(const FDictPoint& ActionMap);

	/**
	 * @brief Collect Observations from the observers
	 * @return The aggregated observations as DictPoint
	 */
	FDictPoint& AggregateObservations();

	/**
	 * @brief Reset the Observers and Actuators managed by this InteractionManager
	 */
	void Reset();
};