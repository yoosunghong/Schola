// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/InferencePolicy.h"

int ConvertFromOneHot(TArray<int> OneHotVector)
{
	// OneHotVector should always contain a 1 otherwise behaviour is undefined
	int Index = -1;
	OneHotVector.Find(1, Index);
	return Index;
}

UInferencePolicy::UInferencePolicy()
{
}

TArray<FString> UInferencePolicy::GetRuntimeNames() const
{
	// we don't support RDG yet so skip it here
	// TArray < FString> ValidRuntimes = UE::NNE::GetAllRuntimeNames < INNERuntimeCPU>();
	// ValidRuntimes.Append(UE::NNE::GetAllRuntimeNames<INNERuntimeGPU>());
	return UE::NNE::GetAllRuntimeNames();
}

IRuntimeInterface* UInferencePolicy::GetRuntime(const FString& SelectedRuntimeName) const
{
	TWeakInterfacePtr<INNERuntimeCPU> CPUPtr = UE::NNE::GetRuntime<INNERuntimeCPU>(SelectedRuntimeName);
	if (CPUPtr.IsValid())
	{
		return new UCPURuntimeWrapper(CPUPtr);
	}

	TWeakInterfacePtr<INNERuntimeGPU> GPUPtr = UE::NNE::GetRuntime<INNERuntimeGPU>(SelectedRuntimeName);
	if (GPUPtr.IsValid())
	{
		return new UGPURuntimeWrapper(GPUPtr);
	}
	// Should probably never happen but
	return nullptr;
}


TFuture<FPolicyDecision*> UInferencePolicy::RequestDecision(const FDictPoint& Observations)
{
	TPromise<FPolicyDecision*>* DecisionPromisePtr = new TPromise<FPolicyDecision*>();
	// Get our future before it can potentially be cleaned up
	TFuture<FPolicyDecision*> FutureDecision = DecisionPromisePtr->GetFuture();

	if (!this->bNetworkLoaded)
	{
		DecisionPromisePtr->EmplaceValue(FPolicyDecision::PolicyError());
		delete DecisionPromisePtr;
	}
	else
	{
		AsyncTask(ENamedThreads::AnyNormalThreadNormalTask, [this, Observations, DecisionPromisePtr]() {
			FPolicyDecision* Decision = new FPolicyDecision(EDecisionType::ACTION);

			for (int i = 0; i < this->ObservationSpaceDefn.Spaces.Num(); i++)
			{
				const TSpace& Space = this->ObservationSpaceDefn.Spaces[i];
				const TPoint& Point = Observations[i];
				TArray<float>& Buffer = this->ObservationBuffer[i].Buffer;
				Visit([&Point, &Buffer](auto& TypedSpace) { TypedSpace.FlattenPoint(Buffer, Point); }, Space);
			}

			// Shift current state by 1
			for (int i = 0; i < StateSeqLen - 1; i++)
			{
				FMemory::Memcpy(StateBuffer.GetData() + i * StateDimSize, StateBuffer.GetData() + (i + 1) * StateDimSize, StateDimSize * sizeof(float));
			}
			if ((int)this->ModelInstance->RunSync(InputBindings, OutputBindings) != 0)
			{
				DecisionPromisePtr->EmplaceValue(FPolicyDecision::PolicyError());
				UE_LOG(LogSchola, Error, TEXT("Failed to run the model"));
			}
			else
			{
				int Index = 0;
				for (const TSpace& Space : this->ActionSpaceDefn.Spaces)
				{
					TArray<float>& Buffer = this->ActionBuffer[Index].Buffer;
					TPoint Point = Visit([&Buffer](auto& TypedSpace) { return TypedSpace.UnflattenAction(Buffer,0); }, Space);
					Decision->Action.Values.Add(Point);
					Index++;
				}
				DecisionPromisePtr->EmplaceValue(Decision);
			}

			delete DecisionPromisePtr;
		});
	}
	return FutureDecision;
}

int FindByName(const FString& Name, TConstArrayView<UE::NNE::FTensorDesc> InArrayView)
{
	for (int i = 0; i < InArrayView.Num(); i++)
	{
		if (Name == InArrayView[i].GetName())
		{
			return i;
		}
	}
	return -1;
}

bool DoOrderResolution(TConstArrayView<UE::NNE::FTensorDesc> InTensorDescs, FInferencePolicyBuffer& InSpaceBuffer, TArray<UE::NNE::FTensorBindingCPU>& OutTensorBindings)
{
	int NextUnMatchedTensor = 0;

	while (NextUnMatchedTensor < InTensorDescs.Num())
	{
		UE::NNE::FTensorBindingCPU& TensorBinding = OutTensorBindings[NextUnMatchedTensor];
	
		if (TensorBinding.Data == nullptr)
		{
			// We have a tensor that is not matched to a space
			int Size = 1;
			for (int i = 1; i < InTensorDescs[NextUnMatchedTensor].GetShape().Rank(); i++)
			{
				Size = Size * InTensorDescs[NextUnMatchedTensor].GetShape().GetData()[i];
			}
			if (Size == InSpaceBuffer.Num())
			{
				TensorBinding = { InSpaceBuffer.GetData(), InSpaceBuffer.Num() * sizeof(float) };
				return true;
			}
		}
		NextUnMatchedTensor++;
	}
	return false;
}

bool UInferencePolicy::SetupBuffersAndBindings(const FInteractionDefinition& PolicyDefinition, TSharedPtr<IModelInstanceRunSync> ModelInstancePtr)
{
	//Setup the Bindings
	this->InputBindings.Init({ nullptr, 0 }, ModelInstancePtr->GetInputTensorDescs().Num());
	this->OutputBindings.Init({ nullptr, 0 }, ModelInstancePtr->GetOutputTensorDescs().Num());
	
	// Setup the Buffers
	int Index = 0;
	for (const TSpace& ObsSpace : PolicyDefinition.ObsSpaceDefn.Spaces)
	{
		this->ObservationBuffer.Emplace();
		Visit([&Index, this](auto& TypedSpace) { this->ObservationBuffer[Index].Init(TypedSpace.GetFlattenedSize()); }, ObsSpace);
		Index++;
	}

	Index=0;
	for (const TSpace& ActionSpace : PolicyDefinition.ActionSpaceDefn.Spaces)
	{
		this->ActionBuffer.Emplace();
		Visit([&Index, this](auto& TypedSpace) { this->ActionBuffer[Index].Init(TypedSpace.GetFlattenedSize()); }, ActionSpace);
		Index++;
	}

	// Create bindings for the observation space
	TArray<int> UnpairedInputs = TArray<int>();
	for (int i = 0; i < PolicyDefinition.ObsSpaceDefn.Spaces.Num(); i++)
	{
		FString Name = PolicyDefinition.ObsSpaceDefn.Labels[i];
		int		IndexInInputTensors = FindByName(Name, ModelInstancePtr->GetInputTensorDescs());
		if (IndexInInputTensors != -1)
		{
			this->InputBindings[IndexInInputTensors] = { this->ObservationBuffer[i].GetData(), this->ObservationBuffer[i].Num() * sizeof(float) };
		}
		else
		{
			UnpairedInputs.Add(i);
		}
	}
	

	for (int UnpairedObsIndex: UnpairedInputs)
	{
		if (!DoOrderResolution(ModelInstancePtr->GetInputTensorDescs(), ObservationBuffer[UnpairedObsIndex], InputBindings))
		{
			// Log that we failed to resolve a tensor for
			UE_LOG(LogSchola, Error, TEXT("Failed to resolve a model input for observation space %s"), *PolicyDefinition.ObsSpaceDefn.Labels[UnpairedObsIndex]);
			return false;
		}
	}
	
	// Create bindings for the action space
	TArray<int> UnPairedOutputs = TArray<int>();
	for (int i = 0; i < PolicyDefinition.ActionSpaceDefn.Spaces.Num(); i++)
	{
		FString Name = PolicyDefinition.ActionSpaceDefn.Labels[i];
		int		IndexInOutputTensors = FindByName(Name, ModelInstancePtr->GetOutputTensorDescs());
		if (IndexInOutputTensors != -1)
		{
			this->OutputBindings[IndexInOutputTensors] = { this->ActionBuffer[i].GetData(), this->ActionBuffer[i].Num() * sizeof(float) };
		}
		else
		{
			UnPairedOutputs.Add(i);
		}
	}

	for (int UnpairedActionIndex : UnPairedOutputs)
	{
		if (!DoOrderResolution(ModelInstancePtr->GetOutputTensorDescs(), ActionBuffer[UnpairedActionIndex], OutputBindings))
		{
			// Log that we failed to resolve a tensor for
			UE_LOG(LogSchola, Error, TEXT("Failed to resolve a model output for action space %s"), *PolicyDefinition.ActionSpaceDefn.Labels[UnpairedActionIndex]);
			return false;
		}
	}

	// Setup State, ignoring if there is no "state_in"/"state_out" tensors
	int InStateIndex = FindByName(FString("state_in"), ModelInstancePtr->GetInputTensorDescs());
	if (InStateIndex != -1)
	{
		UE::NNE::FTensorDesc StateDesc = ModelInstancePtr->GetInputTensorDescs()[InStateIndex];
		if (StateDesc.GetShape().Rank() != 3)
		{
			UE_LOG(LogSchola, Error, TEXT("Invalid input tensor shape for state buffer, should have rank 3"));
			return false;
		}

		StateSeqLen = StateDesc.GetShape().GetData()[1];
		StateDimSize = StateDesc.GetShape().GetData()[2];
		this->StateBuffer.SetNumZeroed(StateSeqLen * StateDimSize);
		InputBindings[InStateIndex] = { StateBuffer.GetData(), StateSeqLen * StateDimSize * sizeof(float) };
		
		// If there is no state input we ignore the state output
		int OutStateIndex = FindByName(FString("state_out"), ModelInstancePtr->GetOutputTensorDescs());
		if (OutStateIndex != -1)
		{
			OutputBindings[OutStateIndex] = { StateBuffer.GetData() + (StateSeqLen - 1) * StateDimSize, StateDimSize * sizeof(float) };
		}
		else
		{
			UE_LOG(LogSchola, Warning, TEXT("No state_out tensor found, ignoring state output"));
		}
	}

	TArray<UE::NNE::FTensorShape> TempShapeArray;

	for (const UE::NNE::FTensorDesc& TensorDesc : ModelInstancePtr->GetInputTensorDescs())
	{
		TempShapeArray.Add(UE::NNE::FTensorShape::MakeFromSymbolic(TensorDesc.GetShape()));
	}
	
	return UE::NNE::IModelInstanceRunSync::ESetInputTensorShapesStatus::Ok == ModelInstancePtr->SetInputTensorShapes(TempShapeArray);
}


void UInferencePolicy::Init(const FInteractionDefinition& PolicyDefinition)
{
	Step = 0;
	ActionSpaceDefn = PolicyDefinition.ActionSpaceDefn;
	ObservationSpaceDefn = PolicyDefinition.ObsSpaceDefn;
	
	if (ModelData)
	{
		TUniquePtr<IRuntimeInterface> Runtime = TUniquePtr<IRuntimeInterface>(this->GetRuntime(this->RuntimeName));
		if (Runtime.IsValid() && Runtime->IsValid())
		{
			TUniquePtr<IModelInterface> TempModelPtr = Runtime->CreateModel(ModelData);
			if (TempModelPtr.IsValid() && TempModelPtr->IsValid())
			{
				ModelInstance = TempModelPtr->CreateModelInstance();
				if (ModelInstance.IsValid())
				{
					if (this->SetupBuffersAndBindings(PolicyDefinition, ModelInstance))
					{
						bNetworkLoaded = true;
					}
					else
					{
						UE_LOG(LogSchola, Error, TEXT("Failed to setup buffers and bindings"));
						bNetworkLoaded = false;
					}
				}
				else
				{
					UE_LOG(LogSchola, Error, TEXT("Failed to create the model instance"));
					bNetworkLoaded = false;
				}
			}
			else
			{
				UE_LOG(LogSchola, Warning, TEXT("Failed to Create the Model"));
				// Invalid Runtime
				bNetworkLoaded = false;
			}
		}
		else
		{
			UE_LOG(LogSchola, Error, TEXT("Cannot find runtime %s, please enable the corresponding plugin"), *this->RuntimeName);
			// Invalid Runtime
			bNetworkLoaded = false;
		}
	}
	else
	{
		UE_LOG(LogSchola, Warning, TEXT("Failed to Create Network Due to Invalid Model Data"));
		// Invalid Model Data
		bNetworkLoaded = false;
	}
}