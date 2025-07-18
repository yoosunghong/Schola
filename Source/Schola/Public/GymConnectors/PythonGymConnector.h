// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "./ExternalGymConnector.h"
#include "Agent/AgentAction.h"
#include "../Generated/GymConnector.pb.h"
#include "../Generated/StateUpdates.pb.h"
#include "../Generated/State.pb.h"
#include "../Generated/GymConnector.grpc.pb.h"
#include "Communicator/CommunicationManager.h"
#include "Communicator/ExchangeRPCBackend.h"
#include "Communicator/PollingRPCBackend.h"
#include "Communicator/ComBackendInterface.h"
#include "GymConnectors/AutoResetTypeEnum.h"
#include "PythonGymConnector.generated.h"

using Schola::InitialTrainingState;
using Schola::InitialTrainingStateRequest;
using Schola::GymService;
using Schola::GymConnectorStartRequest;
using Schola::GymConnectorStartResponse;
using Schola::TrainingDefinition;
using Schola::TrainingDefinitionRequest;
using Schola::TrainingState;
using Schola::TrainingStateUpdate;

/**
 * @brief A struct representing a message indicating that the connector should start.
 */
USTRUCT()
struct FStartRequest
{
	
	GENERATED_BODY()

	/** The AutoReset Type Received from Python. */
	UPROPERTY()
	EAutoResetType AutoResetType = EAutoResetType::SameStep; // Default to SameStep
	
	/**
	 * @brief Create a new empty FStartRequest.
	 */
	FStartRequest();

	/**
	 * @brief Populate an FStartRequest from a proto message.
	 * @param[out] EmptyResetRequest The FStartRequest to populate.
	 * @param[in] ProtoMsg The proto message to populate from.
	 */
	static void FromProto(FStartRequest& EmptyResetRequest, const GymConnectorStartRequest& ProtoMsg);

	/**
	 * @brief Construct an FStartRequest from a proto message.
	 * @param[in] ProtoMsg The proto message to populate from.
	 */
	FStartRequest(const GymConnectorStartRequest& ProtoMsg);

	/**
	 * @brief Create a new FStartRequest from a proto message.
	 * @param[in] ProtoMsg The proto message to populate from.
	 * @return The new FStartRequest.
	 */
	static FStartRequest* FromProto(const GymConnectorStartRequest& ProtoMsg);
};

/**
 * @brief A connection to an external gym API implemented in Python, using gRPC for communication.
 * @note This can theoretically work with any gRPC client, not just Python, although that is untested currently.
 */
UCLASS()
class SCHOLA_API UPythonGymConnector : public UExternalGymConnector
{
	GENERATED_BODY()
private:
	/** A type for an exchange interface that exchanges TrainingState for TrainingStateUpdates. */
	typedef IExchangeBackendInterface<Schola::TrainingStateUpdate, Schola::TrainingState>* DRSType;
	/** A type for a producer interface that sends TrainingDefinition. */
	typedef IProducerBackendInterface<Schola::TrainingDefinition>* ADSType;
	/** A type for a producer interface that publishes InitialTrainingStates after each reset. */
	typedef IProducerBackendInterface<InitialTrainingState>* PRSType;
	/** A type for a consumer interface that collects an initial GymConnectorStartRequest. */
	IPollingBackendInterface<GymConnectorStartRequest>* StartRequestService;

	/** The service that will handle decision requests. */
	DRSType DecisionRequestService;
	/** The service that will handle publishing agent definitions. */
	ADSType AgentDefinitionService;
	/** The service that will handle publishing the state after each reset. */
	PRSType PostResetStateService;

	/** The communication manager that will handle the gRPC server. */
	UPROPERTY()
	UCommunicationManager* CommunicationManager;
	// Initialized in Constructor for now

	UPROPERTY()
	EAutoResetType AutoResetType = EAutoResetType::SameStep; // Default to SameStep

public:
	/**
	 * @brief Create a new UPythonGymConnector.
	 */
	UPythonGymConnector();

	/**
	 * @brief Request a state update from the gym API using the current state of the agents from environments.
	 * @return A future that will eventually contain the decision for all agents in all environments.
	 * @note This function is asynchronous and will return immediately.
	 */
	TFuture<FTrainingStateUpdate*> RequestStateUpdate() override;

	/**
	 * @brief Send the current state to the gym.
	 * @param[in] Value The current state of the training.
	 */
	void SendState(const FTrainingState& Value) override;

	/**
	 * @brief Submit the post-reset state to the gym.
	 * @param[in] Value The state after a reset.
	 */
	void SubmitPostResetState(const FTrainingState& Value) override;

	/**
	 * @brief Initialize the connector with the given training definitions.
	 * @param[in] AgentDefns The training definitions to initialize with.
	 */
	void Init(const FTrainingDefinition& AgentDefns) override;

	/**
	 * @brief Enable the connector.
	 */
	void Enable() override;

	/**
	 * @brief Check if the start request has been received.
	 * @return True if the start request has been received, false otherwise.
	 */
	bool CheckForStart() override;

	EAutoResetType GetAutoResetType() override;


};