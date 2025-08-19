// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Common/Points/Point.h"
#include "Common/Points/PointVisitor.h"
#include "DiscretePoint.generated.h"

USTRUCT(BlueprintType)
struct SCHOLA_API FDiscretePoint : public FPoint
{
	GENERATED_BODY()
	/** the values of this point */
	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category = "Point")
	TArray<int> Values;

	/**
	 * @brief Construct an empty DiscretePoint
	 */
	FDiscretePoint()
	{
	}

	/**
	 * @brief Construct a DiscretePoint from a raw array of ints
	 * @param[in] Data The raw array of ints, as a const ptr
	 * @param[in] Num The size of the array
	 */
	FDiscretePoint(const int* Data, int Num)
		: Values(Data, Num)
	{
	}

	/**
	 * @brief Construct a DiscretePoint from a TArray of ints
	 * @param[in] InitialValues An Array of Ints to initialize the DiscretePoint with
	 */
	FDiscretePoint(TArray<int>& InitialValues)
		: Values(InitialValues)
	{
	}

	/**
	 * @brief Construct a DiscretePoint from a initializer list of ints
	 * @param[in] InitialValues An initializer list of Ints to initialize the DiscretePoint with
	 */
	FDiscretePoint(std::initializer_list<int> InitialValues)
		: Values(InitialValues)
	{
	}

	virtual ~FDiscretePoint()
	{
	}

	void Accept(PointVisitor& Visitor) override;

	void Accept(ConstPointVisitor& Visitor) const override;

	/**
	 * @brief Get the value of the DiscretePoint at the given index or dimension
	 * @param[in] Index The dimension to get the value at
	 * @return the value at the given index or dimension
	 */
	int operator[](int Index) const
	{
		return this->Values[Index];
	}

	/**
	 * @brief Reset the values of the DiscretePoint. Clears the current values
	 * @note This is doesn't reset the size of the array so subsequent calls to Add will not reallocate memory
	 */
	void Reset() override
	{
		this->Values.Reset(this->Values.Num());
	};

	/**
	 * @brief Add a value to the DiscretePoint. Adds a new dimension to the point
	 * @param[in] Value The value to add
	 */
	void Add(int Value)
	{
		this->Values.Add(Value);
	}

	/**
	 * @brief Convert this point to a string representation
	 * @return A string representation of this point
	 */
	FString ToString() const override
	{
		FString Result = TEXT("");
		for (int i = 0; i < this->Values.Num(); i++)
		{
			Result += FString::Printf(TEXT("%d"), this->Values[i]);
			if (i != this->Values.Num() - 1)
			{
				Result += TEXT(", ");
			}
		}
		return Result;
	}
};
