// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Brains/AbstractBrain.h"
#include "Common/InteractionManager.h"
#include "IInferenceAgent.generated.h"

/**
 * @brief The Agent State as represented in Schola. In this case, Stopped means the agent is not taking new instructions.
 */
UENUM(BlueprintType)
enum class EAgentStatus : uint8
{
	Running UMETA(DisplayName = "Running"),
	Stopped UMETA(DisplayName = "Stopped"),
	Error	UMETA(DisplayName = "Error")
};

struct FThinkTickFunction;
struct FActTickFunction;

/**
 * @brief An interface implemented by classes that represent an inference agent.
 */
UINTERFACE(MinimalAPI, Blueprintable)
class UInferenceAgent : public UInterface
{
	GENERATED_BODY()
};

class IInferenceAgent
{
	GENERATED_BODY()

public:
	/**
	 * @brief Get the controlled pawn of the agent.
	 * @return A pointer to a controlled pawn object.
	 */
	virtual APawn* GetControlledPawn() PURE_VIRTUAL(IInferenceAgent::GetControlledPawn, return nullptr;);

	/**
	 * @brief Get the interaction manager for collecting actuators and observations.
	 * @return A pointer to an interaction manager object.
	 */
	virtual UInteractionManager* GetInteractionManager() PURE_VIRTUAL(IInferenceAgent::GetInteractionManager, return nullptr;);

	/**
	 * @brief Get the brain of the agent.
	 * @return A pointer to a brain object.
	 */
	virtual UAbstractBrain* GetBrain() PURE_VIRTUAL(IInferenceAgent::GetBrain, return nullptr;);

	/**
	 * @brief Get the policy of the agent.
	 * @return A pointer to a policy object.
	 */
	virtual UAbstractPolicy* GetPolicy() PURE_VIRTUAL(IInferenceAgent::GetPolicy, return nullptr;);

	/**
	 * @brief Get both the observers attached to the controlled pawn and the observers attached to the agent.
	 * @return An array of observer objects.
	 */
	virtual TArray<UAbstractObserver*> GetAllObservers() PURE_VIRTUAL(IInferenceAgent::GetAllObservers, return TArray<UAbstractObserver*>(););

	/**
	 * @brief Get both the actuators attached to the controlled pawn and the actuators attached to the agent.
	 * @return An array of actuator objects.
	 */
	virtual TArray<UActuator*> GetAllActuators() PURE_VIRTUAL(IInferenceAgent::GetAllActuators, return TArray<UActuator*>(););

	/**
	 * @brief Get the status of the agent.
	 * @return The status of the agent.
	 */
	virtual EAgentStatus GetStatus() PURE_VIRTUAL(IInferenceAgent::GetStatus, return EAgentStatus::Running;);

	/**
	 * @brief Set the status of the agent.
	 * @param NewStatus The new status to set.
	 */
	virtual void SetStatus(EAgentStatus NewStatus) PURE_VIRTUAL(IInferenceAgent::SetStatus, return;);

	/**
	 * @brief Get all observers attached to a pawn.
	 * @return An array of observer objects.
	 */
	TArray<UAbstractObserver*> GetObserversFromPawn();

	/**
	 * @brief Get all actuators attached to a pawn.
	 * @return An array of actuator objects.
	 */
	TArray<UActuator*> GetActuatorsFromPawn();

	/**
	 * @brief Get the name of the agent.
	 * @return The name of the agent.
	 */
	FString GetAgentName();

	/**
	 * @brief Initialize this agent after play has begun.
	 * @return True if initialization was successful, false otherwise.
	 */
	bool Initialize();

	/**
	 * @brief The agent retrieves an action from the brain before taking an action.
	 */
	virtual void Act();

	/**
	 * @brief Update the state of the agent. This checks if the agent is done, what its reward should be, and does any observation collection before requesting a decision.
	 */
	virtual void Think();

	/**
	 * @brief Register default tick functions for Think and Act.
	 */
	virtual void SetupDefaultTicking(FThinkTickFunction& OutThinkTickFunction, FActTickFunction& OutActTickFunction, AActor* InTargetActor = nullptr);
};

/**
 * @brief A struct that represents the think portion of the tick function for the agent.
 */
USTRUCT()
struct FThinkTickFunction : public FTickFunction
{
	GENERATED_BODY()

	/** The agent associated with this tick function. */
	UPROPERTY()
	TScriptInterface<IInferenceAgent> Agent;

	/**
	 * @brief Constructor with agent parameter.
	 * @param Agent The agent associated with this tick function.
	 */
	FThinkTickFunction(IInferenceAgent* Agent);

	FThinkTickFunction()
		: Super()
	{
	}

	/**
	 * @brief Execute the tick function.
	 * @param[in] DeltaTime The time since the last tick.
	 * @param[in] TickType The type of tick.
	 * @param[in] CurrentThread The current thread.
	 * @param[in] MyCompletionGraphEvent The completion graph event.
	 */
	void ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent) override;

	/** Abstract function to describe this tick. Used to print messages about illegal cycles in the dependency graph **/
	virtual FString DiagnosticMessage() override;
	virtual FName	DiagnosticContext(bool bDetailed) override;
};

template <>
struct TStructOpsTypeTraits<FThinkTickFunction> : public TStructOpsTypeTraitsBase2<FThinkTickFunction>
{
	enum
	{
		WithCopy = false,
		WithPureVirtual = false
	};
};

/**
 * @brief A struct that represents the act portion of the tick function for the agent.
 */
USTRUCT()
struct FActTickFunction : public FTickFunction
{
	GENERATED_BODY()

	/** The agent associated with this tick function. */
	UPROPERTY()
	TScriptInterface<IInferenceAgent> Agent;

	/** Whether to stop after the current tick. */
	UPROPERTY()
	bool bStopAfterCurrentTick = false;

	/**
	 * @brief Constructor with agent and stop flag parameters.
	 * @param[in] Agent The agent associated with this tick function.
	 * @param[in] bStopAfterCurrentTick Whether to stop after the current tick.
	 */
	FActTickFunction(IInferenceAgent* Agent, bool bStopAfterCurrentTick = false);

	FActTickFunction()
		: Super()
	{
	}

	/**
	 * @brief Execute the tick function.
	 * @param[in] DeltaTime The time since the last tick.
	 * @param[in] TickType The type of tick.
	 * @param[in] CurrentThread The current thread.
	 * @param[in] MyCompletionGraphEvent The completion graph event.
	 */
	void ExecuteTick(float DeltaTime, ELevelTick TickType, ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent) override;

	/** Abstract function to describe this tick. Used to print messages about illegal cycles in the dependency graph **/
	virtual FString DiagnosticMessage() override;
	virtual FName	DiagnosticContext(bool bDetailed) override;
};

template <>
struct TStructOpsTypeTraits<FActTickFunction> : public TStructOpsTypeTraitsBase2<FActTickFunction>
{
	enum
	{
		WithCopy = false,
		WithPureVirtual = false
	};
};