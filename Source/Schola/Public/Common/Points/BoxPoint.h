// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Points/Point.h"
#include "Common/Points/PointVisitor.h"
#include "BoxPoint.generated.h"

/**
 * @brief A point in a box(continuous) space. Conceptually a floating point vector
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FBoxPoint : public FPoint
{
	GENERATED_BODY()

	/** the values of this point */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	TArray<float> Values;

	/**
	 * @brief Construct an empty BoxPoint
	 */
	FBoxPoint()
	{
	}

	FBoxPoint(const TArray<float>& InValues)
		: Values(InValues)
	{
	}

	FBoxPoint(std::initializer_list<float> InValues)
		: Values(InValues)
	{
	}

	/**
	 * @brief Construct a BoxPoint from a raw array of floats
	 * @param[in] Data The raw array of floats, as a const ptr
	 * @param[in] Num The size of the array
	 */
	FBoxPoint(const float* Data, int Num)
		: Values(Data, Num)
	{
	}

	/**
	 * @brief Construct a preallocated BoxPoint, with no values
	 * @param[in] NumDims The amount of memory to preallocate
	 */
	FBoxPoint(int NumDims)
	{
		Values.Reserve(NumDims);
	}

	/**
	 * @brief Get the value of the BoxPoint at the given index or dimension
	 * @param Index The dimension to get the value at
	 * @return the value at the given index or dimension
	 */
	float operator[](int Index) const
	{
		return this->Values[Index];
	}

	virtual ~FBoxPoint()
	{
	}

	/**
	 * @brief Add a value to the BoxPoint. Adds a new dimension to the point
	 * @param[in] Value The value to add
	 */
	void Add(float Value)
	{
		this->Values.Add(Value);
	}
	/**
	 * @brief Reset the values of the BoxPoint. Clears the current values
	 * @note This is doesn't reset the size of the array so subsequent calls to Add will not reallocate memory
	 */
	void Reset() override
	{
		this->Values.Reset(this->Values.Num());
	};

	void Accept(PointVisitor& Visitor) override;

	void Accept(ConstPointVisitor& Visitor) const override;

	/**
	 * @brief Convert this point to a string representation
	 * @return A string representation of this point
	 */
	FString ToString() const override
	{
		FString Result = TEXT("");
		for (int i = 0; i < this->Values.Num(); i++)
		{
			Result += FString::SanitizeFloat(this->Values[i]);
			if (i != this->Values.Num() - 1)
			{
				Result += TEXT(", ");
			}
		}
		return Result;
	};
};