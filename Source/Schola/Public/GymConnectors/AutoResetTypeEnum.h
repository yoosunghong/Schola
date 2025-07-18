// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "AutoResetTypeEnum.generated.h"

UENUM()
enum class EAutoResetType
{
	Disabled UMETA(DisplayName = "Disabled"), // Only reset the environment when we receive a reset message.
	NextStep UMETA(DisplayName = "NextStep"),  // Not Supported. Reset the environment with the first step (so we discard an action)
	SameStep UMETA(DisplayName = "SameStep"),  // Reset the environment, mid-step and put the last obs in the info dictionary.
};
