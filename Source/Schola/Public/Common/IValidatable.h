// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "UObject/Interface.h"
#include "IValidatable.generated.h"

UENUM(BlueprintType)
enum class EValidationType : uint8
{
	NONE UMETA(DisplayName = "No Validation"),
	WARN UMETA(DisplayName = "Warn"),
	FAIL UMETA(DisplayName = "Error"),
};

/**
 * @brief Enum class of validation results.
 */
UENUM(BlueprintType)
enum class EValidationResult : uint8
{
	PASS UMETA(DisplayName = "Pass"),
	WARN UMETA(DisplayName = "Warn"),
	FAIL UMETA(DisplayName = "Error")
};

/**
 * @brief Enum class of validation results for spaces.
 */
UENUM(BlueprintType)
enum class ESpaceValidationResult : uint8
{
	NoResults		UMETA(DisplayName = "No Results"),
	Success			UMETA(DisplayName = "Success"),
	WrongDimensions UMETA(DisplayName = "Wrong Dimensions"),
	OutOfBounds		UMETA(DisplayName = "Value Out of Bounds"),
	WrongDataType   UMETA(DisplayName = "Data was of wrong type")
};

/**
 * @brief Struct contraining validation result and its causer if unsuccessful.
 */
USTRUCT()
struct FValidationResult
{
	GENERATED_BODY()

	/** The result of running validation  */
	UPROPERTY()
	ESpaceValidationResult Result = ESpaceValidationResult::NoResults;

	/** The object that caused the validation to fail */
	UPROPERTY()
	UObject* Cause;

	FValidationResult()
	{
		Cause = nullptr;
	}

	FValidationResult(ESpaceValidationResult Result, UObject* Cause)
		: Result(Result), Cause(Cause)
	{
	}
};

/** 
 * @brief Helper function to convert EValidationResult into bool based on ValidationType
 * @param[in] Result The result to convert
 * @param[in] ValidationType The type of validation to perform when converting (e.g. do we warn or fail)
 * @return true if the validation was successful, false otherwise
 */
inline bool ConvertValidatationResult(const EValidationResult Result, const EValidationType ValidationType = EValidationType::FAIL)
{
	// Successful in 3 cases:n
	// 1) Validation Type is "No Validation"
	// 2) Validation Type is "Warn", and Validation Result is "Pass" or "Warn"
	// 3) Validation Type is "Fail", and Validation Result is "Pass"
	return ValidationType == EValidationType::NONE || (ValidationType == EValidationType::WARN && Result != EValidationResult::FAIL) || Result == EValidationResult::PASS;
};


/** 
 * @brief Helper function to convert ESpaceValidationResult into EValidationResult 
 * @param[in] Result The result to convert
 * @return the converted result
 */
inline EValidationResult ConvertSpaceValidatationResult(const FValidationResult Result)
{
	if (Result.Result == ESpaceValidationResult::Success)
	{
		return EValidationResult::PASS;
	}

	else if (Result.Result == ESpaceValidationResult::OutOfBounds)
	{
		return EValidationResult::WARN;
	}

	else if (Result.Result == ESpaceValidationResult::WrongDimensions)
	{
		return EValidationResult::FAIL;
	}

	else
	{
		return EValidationResult::FAIL;
	}
};

/**
 * @brief A class that can be validated.
 * @details validatable validates itself by checking that its and its components, ensuring that spaces of values collected from the environment matches with space definitions.
 */
UINTERFACE(MinimalAPI, Blueprintable)
class UValidatable : public UInterface
{
	GENERATED_BODY()
};

class IValidatable
{
	GENERATED_BODY()

public:

	/**
	 * @brief Validate the object
	 * @param[in] ValidationType The type of validation to perform when converting (e.g. do we warn or fail)
	 */
	virtual bool Validate(EValidationType ValidationType = EValidationType::FAIL) = 0;

	/**
	 * @brief Callback to perform when validation results in a warning.
	 * @param[in] WarnResult The result of the validation
	 * @note implement this function to handle validation warnings
	 */
	virtual void OnWarn(FValidationResult WarnResult){};

	/**
	 * @brief Callback to perform when validation results in a Failure.
	 * @param[in] FailResult The result of the validation
	 * @note implement this function to handle validation failures
	 */
	virtual void OnFail(FValidationResult FailResult){};
	/**
	 * @brief Callback to perform when validation results in a Pass.
	 * @param[in] SuccessResult The result of the validation
	 * @note implement this function to handle validation passes
	 */
	virtual void OnPass(FValidationResult SuccessResult){};

	/**
	 * @brief Generate a validation result from a space validation result
	 * @param[in] Result The space validation result
	 * @return the converted validation result
	 * @note This function will call the appropriate callback based on the result
	 */
	EValidationResult GenerateValidatationResult(const FValidationResult Result)
	{
		EValidationResult ConvertedResult = ConvertSpaceValidatationResult(Result);
		if (ConvertedResult == EValidationResult::PASS)
		{
			OnPass(Result);
		}

		else if (ConvertedResult == EValidationResult::WARN)
		{
			OnWarn(Result);
		}

		else if (ConvertedResult == EValidationResult::FAIL)
		{
			OnFail(Result);
		}

		else
		{
			//Do Nothing
		}

		return ConvertedResult;
	}

	/** 
	 * @brief Generate a validation result from a space validation result, without a source object
	 * @param[in] Result The space validation result
	 * @return the converted validation result
	*/
	EValidationResult GenerateValidatationResult(const ESpaceValidationResult Result)
	{
		return GenerateValidatationResult(FValidationResult(Result, nullptr));
	}
};