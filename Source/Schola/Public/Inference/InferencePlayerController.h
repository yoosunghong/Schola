// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include <Kismet/GameplayStatics.h>
#include "Common/LogSchola.h"
#include "./IInferenceAgent.h"
#include "InferencePlayerController.generated.h"

/**
 * @brief A controller that implements the IInferenceAgent interface, to control a Pawn with a Brain/Policy.
 */
UCLASS(Abstract, Blueprintable, ClassGroup = (Schola), meta = (BlueprintSpawnableComponent))
class SCHOLA_API AInferencePlayerController : public APlayerController, public IInferenceAgent
{
	GENERATED_BODY()

public:
	/** Object defining how the agent interacts with the environment. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UInteractionManager* InteractionManager = CreateDefaultSubobject<UInteractionManager>(TEXT("InteractionManager"));

	/** Object defining an asynchronous function f:Observations->Actions used to make decisions for the agent. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UAbstractPolicy* Policy;

	/** Object defining how decision requests are synchronized. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	UAbstractBrain* Brain;

	/** List of observers that collect observations for the agent. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	TArray<UAbstractObserver*> Observers;

	/** List of actuators that execute actions for the agent. */
	UPROPERTY(EditAnywhere, NoClear, Instanced, meta = (ShowInnerProperties), Category = "Reinforcement Learning")
	TArray<UActuator*> Actuators;

	/** The status of the agent. */
	UPROPERTY(BlueprintReadOnly)
	EAgentStatus Status = EAgentStatus::Stopped;

	/* Tick function for Think calls. */
	UPROPERTY()
	FThinkTickFunction ThinkTickFunction = FThinkTickFunction(this);

	/* Tick function for Act calls. */
	UPROPERTY()
	FActTickFunction ActTickFunction = FActTickFunction(this);

	/** Whether the agent should be set up to take actions automatically. */
	UPROPERTY(EditAnywhere, Category = "Reinforcement Learning")
	bool bRegisterAgentStep = true;

	/** Number of discrete actions for discrete actuators of the agent. */
	UPROPERTY(BlueprintReadOnly, Category = "Reinforcement Learning")
	TArray<int> NumDiscreteActions = {};

	/** Number of continuous actions for continuous actuators of the agent. */
	UPROPERTY(BlueprintReadOnly, Category = "Reinforcement Learning")
	TArray<int> NumContinuousActions = {};

	/** Number of binary actions for binary actuators of the agent. */
	UPROPERTY(BlueprintReadOnly, Category = "Reinforcement Learning")
	TArray<int> NumBinaryActions = {};

	/**
	 * @brief Get the controlled pawn of the agent.
	 * @return A pointer to the controlled pawn.
	 */
	virtual APawn*
	GetControlledPawn() override
	{
		return this->GetPawn();
	}

	/**
	 * @brief Get the interaction manager for the agent.
	 * @return A pointer to the interaction manager.
	 */
	virtual UInteractionManager* GetInteractionManager() override
	{
		return InteractionManager;
	}

	/**
	 * @brief Get the brain of the agent.
	 * @return A pointer to the brain.
	 */
	virtual UAbstractBrain* GetBrain() override
	{
		return Brain;
	}

	/**
	 * @brief Get the policy of the agent.
	 * @return A pointer to the policy.
	 */
	virtual UAbstractPolicy* GetPolicy() override
	{
		return Policy;
	}

	/**
	 * @brief Get all observers of the agent.
	 * @return An array of pointers to the observers.
	 */
	virtual TArray<UAbstractObserver*> GetAllObservers() override
	{
		TArray<UAbstractObserver*> AllObservers;
		AllObservers.Append(this->Observers);
		AllObservers.Append(GetObserversFromPawn());

		TArray<USensor*> SensorsTemp;
		this->GetComponents(SensorsTemp);
		for (USensor* Sensor : SensorsTemp)
		{
			AllObservers.Add(Sensor->Observer);
		}

		return AllObservers;
	}

	/**
	 * @brief Get all actuators of the agent.
	 * @return An array of pointers to the actuators.
	 */
	virtual TArray<UActuator*> GetAllActuators() override
	{

		TArray<UActuator*> AllActuators;
		AllActuators.Append(Actuators);
		AllActuators.Append(GetActuatorsFromPawn());

		TArray<UActuatorComponent*> ActuatorsTemp;
		this->GetComponents(ActuatorsTemp);
		for (UActuatorComponent* Actuator : ActuatorsTemp)
		{
			AllActuators.Add(Actuator->Actuator);
		}

		return AllActuators;
	}

	/**
	 * @brief Get the status of the agent.
	 * @return The status of the agent.
	 */
	virtual EAgentStatus GetStatus() override
	{
		return Status;
	}

	/**
	 * @brief Set the status of the agent.
	 * @param[in] NewStatus The new status to set.
	 */
	virtual void SetStatus(EAgentStatus NewStatus) override
	{
		Status = NewStatus;
	}

	/**
	 * @brief Register or unregister the tick functions for the agent.
	 * @param[in] bRegister Whether to register the tick functions.
	 */
	void RegisterActorTickFunctions(bool bRegister) override
	{
		Super::RegisterActorTickFunctions(bRegister);
		if (bRegister && bRegisterAgentStep)
		{
			// Pass in this as the target actor since the object can exist outside of the controlled actor
			this->SetupDefaultTicking(this->ThinkTickFunction, this->ActTickFunction, this);
		}
		else
		{
			this->ThinkTickFunction.UnRegisterTickFunction();
			this->ActTickFunction.UnRegisterTickFunction();
		}
	}

	/**
	 * @brief Called when the game starts or when spawned.
	 */
	virtual void BeginPlay() override
	{
		Super::BeginPlay();
		this->Initialize();

		for (TSpace& Space : GetInteractionManager()->InteractionDefn.ActionSpaceDefn.Spaces)
		{
			if (Space.IsType<FDiscreteSpace>())
			{
				NumDiscreteActions.Add(Space.Get<FDiscreteSpace>().High.Num());
			}
			else if (Space.IsType<FBoxSpace>())
			{
				NumContinuousActions.Add(Space.Get<FBoxSpace>().Dimensions.Num());
			}
			else if (Space.IsType<FBinarySpace>())
			{
				NumBinaryActions.Add(Space.Get<FBinarySpace>().Shape);
			}
		}
	}

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void DoDiscreteAction(FDiscretePoint& Action);

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void DoContinuousAction(FBoxPoint& Action);

	UFUNCTION(BlueprintImplementableEvent, Category = "Reinforcement Learning")
	void DoBinaryAction(FBinaryPoint& Action);

	// Fill the actuator with the actions at its corresponding index
	void DoDiscreteAction(TArray<TPoint>& Action, TArray<int>& NumActions)
	{
		int			   ActionStartIndex = 0;
		FDiscretePoint ActionPoint = FDiscretePoint{};
		this->DoDiscreteAction(ActionPoint);
		for (int i = 0; i < NumActions.Num(); i++)
		{

			FDiscretePoint ActionPointBranched;
			for (int j = ActionStartIndex; j < ActionStartIndex + NumActions[i]; j++)
			{
				ActionPointBranched.Add(ActionPoint[j]);
			}
			Action[i].Set<FDiscretePoint>(ActionPointBranched);
			ActionStartIndex += NumActions[i];
		}
	}

	void DoContinuousAction(TArray<TPoint>& Action, TArray<int>& NumActions)
	{
		int ActionStartIndex = 0;
		for (int i = 0; i < NumActions.Num(); i++)
		{
			FBoxPoint ActionPoint = FBoxPoint{};
			this->DoContinuousAction(ActionPoint);
			FBoxPoint ActionPointBranched;
			for (int j = ActionStartIndex; j < ActionStartIndex + NumActions[i]; j++)
			{
				ActionPointBranched.Add(ActionPoint[j]);
			}
			Action[i].Set<FBoxPoint>(ActionPointBranched);
			ActionStartIndex += NumActions[i];
		}
	}

	void DoBinaryAction(TArray<TPoint>& Action, TArray<int>& NumActions)
	{
		int ActionStartIndex = 0;
		for (int i = 0; i < NumActions.Num(); i++)
		{
			FBinaryPoint ActionPoint = FBinaryPoint{};
			this->DoBinaryAction(ActionPoint);
			FBinaryPoint ActionPointBranched;
			for (int j = ActionStartIndex; j < ActionStartIndex + NumActions[i]; j++)
			{
				ActionPointBranched.Add(ActionPoint[j]);
			}
			Action[i].Set<FBinaryPoint>(ActionPointBranched);
			ActionStartIndex += NumActions[i];
		}
	}

	void Think() override
	{
		// Merely get the observations from the interaction manager which will dump to json
		FDictPoint& Obs = GetInteractionManager()->AggregateObservations();
	}

	void Act() override
	{
		// Create an empty action map to hold the actions
		FDictPoint ActionMap = FDictPoint{};

		// Create an array of TPoint, one for each element in NumDiscreteActions
		TArray<TPoint> DiscreteActions;
		DiscreteActions.SetNum(NumDiscreteActions.Num());
		this->DoDiscreteAction(DiscreteActions, NumDiscreteActions);
		for (int i = 0; i < NumDiscreteActions.Num(); i++)
		{
			ActionMap.Add(DiscreteActions[i]);
		}

		TArray<TPoint> ContinuousActions;
		ContinuousActions.SetNum(NumContinuousActions.Num());
		this->DoContinuousAction(ContinuousActions, NumContinuousActions);
		for (int i = 0; i < NumContinuousActions.Num(); i++)
		{
			ActionMap.Add(ContinuousActions[i]);
		}

		TArray<TPoint> BinaryActions;
		BinaryActions.SetNum(NumBinaryActions.Num());
		this->DoBinaryAction(BinaryActions, NumBinaryActions);
		for (int i = 0; i < NumBinaryActions.Num(); i++)
		{
			ActionMap.Add(BinaryActions[i]);
		}

		GetInteractionManager()->DistributeActions(ActionMap);
	}
};