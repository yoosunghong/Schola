// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Points/Point.h"
#include "Common/Points/PointVisitor.h"
#include "BinaryPoint.generated.h"

USTRUCT(BlueprintType)
struct SCHOLA_API FBinaryPoint : public FPoint
{
	GENERATED_BODY()
	/** the values of this point */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	TArray<bool> Values;
	/**
	 * @brief Construct an empty BinaryPoint
	 */
	FBinaryPoint()
	{
	}

	/**
	 * @brief Construct a BinaryPoint from a TArray of bools
	 * @param[in] InitialValues An Array of Bools to initialize the BinaryPoint with
	 */
	FBinaryPoint(TArray<bool>& InitialValues)
		: Values(InitialValues)
	{
	}

	/**
	 * @brief Construct a BinaryPoint from a raw array of bools
	 * @param[in] Data The raw array of bools, as a const ptr
	 * @param[in] Num The size of the array
	 */
	FBinaryPoint(const bool* Data, int Num)
		: Values(Data, Num)
	{
	}

	virtual ~FBinaryPoint() {};
	/**
	 * @brief Get the value of the BinaryPoint at the given index or dimension
	 * @param[in] Index The dimension to get the value at
	 * @return the value at the given index or dimension
	 */
	bool operator[](int Index) const
	{
		return this->Values[Index];
	}

	/**
	 * @brief Add a value to the BinaryPoint. Adds a new dimension to the point
	 * @param[in] Value The value to add
	 */
	void Add(bool Value)
	{
		this->Values.Add(Value);
	}

	/**
	 * @brief Reset the values of the BinaryPoint. Clears the current values
	 * @note This is doesn't reset the size of the array so subsequent calls to Add will not reallocate memory
	 */
	void Reset() override
	{
		this->Values.Reset(Values.Num());
	}

	void Accept(PointVisitor& Visitor) override;

	void Accept(ConstPointVisitor& Visitor) const override;

	/**
	 * @brief Convert this point to a string representation
	 * @return A string representation of this point
	 */

	FString ToString() const override
	{
		FString Result = TEXT("BinaryPoint: ");
		for (int i = 0; i < this->Values.Num(); i++)
		{
			Result += FString::Printf(TEXT("%d "), this->Values[i]);
		}
		return Result;
	}
};