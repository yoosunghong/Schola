// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Common/Points/BinaryPoint.h"

void FBinaryPoint::Accept(ConstPointVisitor& Visitor) const
{
	Visitor.Visit(*this);
}

void FBinaryPoint::Accept(PointVisitor& Visitor)
{
	Visitor.Visit(*this);
}