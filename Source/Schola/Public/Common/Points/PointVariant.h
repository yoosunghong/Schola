// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Misc/TVariant.h"
#include "Common/Points/BinaryPoint.h"
#include "Common/Points/BoxPoint.h"
#include "Common/Points/DiscretePoint.h"

/** A variant of the three concrete point types */
typedef TVariant<FBoxPoint, FBinaryPoint, FDiscretePoint> TPoint;