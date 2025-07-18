// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Points/Point.h"
#include "Common/Points/PointVariant.h"
#include "DictPoint.generated.h"

/**
 * @brief A dictionary of points. This is used to store multiple points, indexed by an integer key
 */
USTRUCT()
struct FDictPoint
{
	GENERATED_BODY()

	/** The points in the dictionary */
	TArray<TPoint> Points;

	/**
	 * @brief Reset all the points in the dictionary
	 */
	void Reset();

	/**
	 * @brief Construct an empty dictionary of points
	 * @note We need to allocate the Buffer first before we start handing out array views which means we can't do it here
	 */
	FDictPoint()
	{

	}

	/**
	 * @brief Add a point to the dictionary, by emplacing it, and returning it to the caller for initialization
	 * @return a reference to the newly added point
	 */
	TPoint& Add()
	{
		return this->Points.Emplace_GetRef();
	}
	/**
	 * @brief Add a preallocated point to the dictionary
	 * @param[in] Point The point to add
	 */
	void Add(TPoint& Point)
	{
		this->Points.Add(Point);
	}

	/**
	 * @brief Get the point at the given Index, but const 
	 * @param Index The index of the point to get
	 * @return a reference to the point at the given Index
	 */
	TPoint& operator[](int Index)
	{
		return this->Points[Index];
	};
	/**
	 * @brief Get the point at the given Index, in a const context
	 * @param Index The index of the point to get
	 * @return a const reference to the point at the given Index
	 */
	const TPoint& operator[](int Index) const
	{
		return this->Points[Index];
	};

	void Accept(PointVisitor& Visitor)
	{
		for (TPoint& Point : this->Points)
		{
			Visit([&Visitor](auto& PointArg) { PointArg.Accept(Visitor); }, Point);
		}
	}

	void Accept(ConstPointVisitor& Visitor) const
	{
		for (TPoint Point : this->Points)
		{
			Visit([&Visitor](const auto& PointArg) { PointArg.Accept(Visitor); }, Point);
		}
	}

	void Accept(PointVisitor* Visitor)
	{
		return this->Accept(*Visitor);
	}

	void Accept(ConstPointVisitor* Visitor) const
	{
		return this->Accept(*Visitor);
	}

};