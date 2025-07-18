// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Agent/AgentUIDSubsystem.h"

void UAgentUIDSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	this->CurrentId = 0;
}

void UAgentUIDSubsystem::Deinitialize()
{
	// Do Nothing
}

int UAgentUIDSubsystem::GetId()
{
	return CurrentId++;
}
