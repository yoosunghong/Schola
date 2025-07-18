// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Async/Future.h"
#include "Common/InteractionDefinition.h"
#include "Agent/AgentAction.h"
#include "Common/Spaces.h"
#include "PolicyDecision.h"
#include "AbstractPolicy.generated.h"

/**
 * An abstract class representing the decision making protocol of an NPC. Without any synchronization.
 */
UCLASS(Abstract, EditInlineNew)
class SCHOLA_API UAbstractPolicy : public UObject
{
	GENERATED_BODY()
public:
	/**
	 * @brief Request that the policy decide on action. May take some time to occur
	 * @param[in] Observations The current state of the agent used to inform the policies choice of action
	 * @return A future that will eventually contain the policy's next decision
	 */
	virtual TFuture<FPolicyDecision*> RequestDecision(const FDictPoint& Observations) PURE_VIRTUAL(UAbstractPolicy::RequestDecision, return TFuture<FPolicyDecision*>(););

	/**
	 * @brief Initialize an instance of a policy object from an interaction definition
	 * @param[in] PolicyDefinition An object defining the policy's I/O shapes and other parameters
	 */
	virtual void Init(const FInteractionDefinition& PolicyDefinition) PURE_VIRTUAL(UAbstractPolicy::Init, return; );
};