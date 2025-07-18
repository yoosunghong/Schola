// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Subsystem/ScholaManagerSubsystem.h"

void UScholaManagerSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	GetWorld()->OnWorldBeginPlay.AddUObject(this, &UScholaManagerSubsystem::PrepareSubsystem);
}

void UScholaManagerSubsystem::Deinitialize()
{
	Super::Deinitialize();
	bSubsystemPrepared = false;
}

void UScholaManagerSubsystem::Tick(float DeltaTime)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Subsystem Tick");

	if (this->GymConnector && this->GymConnector->IsNotStarted())
	{
		bFirstStep = true;
		bool bStarted = this->GymConnector->CheckForStart();
	}

	// Action Phase: We take any actions or Reset the Environment
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Agents Acting");
		if (this->GymConnector && this->GymConnector->IsRunning())
		{
			FTrainingStateUpdate* StateUpdate = this->GymConnector->ResolveEnvironmentStateUpdate();
			// Maybe there was nothing to resolve
			if (StateUpdate)
			{
				this->GymConnector->UpdateConnectorStatus(*StateUpdate);
				this->GymConnector->UpdateEnvironments(*StateUpdate);
			}
		}

		// Reset the environments, if the environment received a reset request
		// Do it after we take inference actions so that if they are linked to the envs they get reset properly
		if (this->GymConnector != nullptr && this->GymConnector->IsRunning())
		{
			
			this->GymConnector->ResetCompletedEnvironments();
		}
	}

	// Thinking Phase: Send the Last State Update to Gym
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Agents Thinking");
		if (this->GymConnector && this->GymConnector->IsRunning())
		{
			
			this->GymConnector->CollectEnvironmentStates();
			this->GymConnector->SubmitEnvironmentStates();
		}
	}

	// self-Reset Phase if we have already run for 1+ steps
	if (this->GymConnector && !bFirstStep && this->GymConnector->IsRunning())
	{
		this->GymConnector->AutoReset();
	}

	bFirstStep = false;
}

ETickableTickType UScholaManagerSubsystem::GetTickableTickType() const
{
	return Super::GetTickableTickType();
}

// Pipe the UObject GetStatID to the abstract method GetStatId in UTickableWorldSubsystem
TStatId UScholaManagerSubsystem::GetStatId() const
{
	return this->GetStatID();
}

void UScholaManagerSubsystem::PrepareSubsystem()
{

	const UScholaManagerSubsystemSettings* ScholaSettings = GetDefault<UScholaManagerSubsystemSettings>();

	// Don't generate a new gym connector if it doesn't exist
	if (*ScholaSettings->GymConnectorClass != nullptr)
	{
		this->GymConnector = NewObject<UAbstractGymConnector>(this, (UClass*) ScholaSettings->GymConnectorClass, FName("GymConnector"));
		this->GymConnector->Init();
	}

	// Setup the EnvController
	int NumAgents = 0;
	if (this->GymConnector)
	{
		for (AAbstractScholaEnvironment* Environment : GymConnector->Environments)
		{
			NumAgents += Environment->GetNumAgents();
		}
	};

	if (this->GymConnector && NumAgents > 0)
	{
		this->GymConnector->Enable();

		UE_LOG(LogSchola, Warning, TEXT("Backend Started"));
		// Make the tick start doing it's thing
		bSubsystemPrepared = true;
		bFirstStep = true;

		// Use the config setting, but we can override the config value by passing ScholaDisableScript on the command line
		if (ScholaSettings->bRunScriptOnPlay && !FParse::Param(FCommandLine::Get(), TEXT("ScholaDisableScript")))
		{
			FLaunchableScript TrainingScript = ScholaSettings->GetScript();
			TrainingScript.LaunchScript();
		}
	}
	else if (NumAgents == 0)
	{
		UE_LOG(LogSchola, Log, TEXT("Nothing found to train, skipping script and GymConnector start"))
	}
}


bool UScholaManagerSubsystem::IsTickable() const
{
	return Super::IsAllowedToTick();
}
