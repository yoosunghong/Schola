// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Point.generated.h"

class PointVisitor;
class ConstPointVisitor;
/**
 * @brief A point is a data point that can be passed to a model for inference. This base class just provides support for the visitor pattern
 */
USTRUCT()
struct SCHOLA_API FPoint
{
	GENERATED_BODY()

	// Array View referencing other memory somewhere.
	// This should store data exactly as it would be passed to Inference

	virtual void Accept(PointVisitor& Visitor) PURE_VIRTUAL(FPoint::Accept, return;);

	virtual void Accept(ConstPointVisitor& Visitor) const PURE_VIRTUAL(FPoint::Accept, return;);

	virtual void Reset() PURE_VIRTUAL(FPoint::Reset, return;);

	/**
	 * @brief Convert this point to a string representation
	 * @return A string representation of this point
	 */
	virtual FString ToString() const PURE_VIRTUAL(FPoint::ToString, return TEXT("Invalid Point"););

	virtual ~FPoint() = default;
};
