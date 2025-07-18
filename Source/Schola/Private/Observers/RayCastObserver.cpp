// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Observers/RayCastObserver.h"

FBoxSpace URayCastObserver::GetObservationSpace() const
{
	FBoxSpace SpaceDefinition;

	for (int i = 0; i < NumRays; i++)
	{
		// First # of tag entries correspond to each tag
		for (const FName& Tag : TrackedTags)
		{
			SpaceDefinition.Dimensions.Add(FBoxSpaceDimension(0.0, 1.0));
		}
		// Did we hit anything at all
		SpaceDefinition.Dimensions.Add(FBoxSpaceDimension(0.0, 1.0));
		// How far away was the thing we hit
		SpaceDefinition.Dimensions.Add(FBoxSpaceDimension(0.0, 1.0));
	}

	return SpaceDefinition;
}

TArray<FVector> URayCastObserver::GenerateRayEndpoints(int32 InNumRays, float InRayDegrees, FVector InBaseEnd, FVector InStart, FTransform InBaseTransform, FVector InEndOffset)
{

	TArray<FVector> OutAngles;
	OutAngles.Init(FVector(), InNumRays);

	float Delta;
	// Special case to avoid 2 rays ontop of each other or a divide by zero
	if (InRayDegrees >= 360.0 || InNumRays <= 1)
	{
		// For 360 degrees, we should have 2 rays -> 180 deg, 3 rays -> 120 deg, 4 rays -> 90 deg, 5 rays -> 72 deg
		Delta = InRayDegrees / InNumRays;
	}
	else
	{
		// Normal Case where we put rays up to the edges of the range
		// For 90 degrees, we should have 2 rays -> 90 deg, 3 rays -> 45 deg, 4 rays -> 30 deg, 5 rays -> 17.5 deg
		Delta = InRayDegrees / (InNumRays - 1);
	}

	for (int32 Index = 0; Index < InNumRays; Index += 1)
	{
		// We want to start on the far left and move to the far right so for 90 degrees first ray is at -45 degrees.
		FRotator Rotator(0.0f, Delta * Index - (InRayDegrees / 2), 0);
		OutAngles[Index] = Rotator.RotateVector(InBaseEnd);
	}

	// Now we have a semi-sphere of points, centered around 0.
	// Get the pawn's transform to apply the base transform relative to the pawn's orientation
	AActor* Owner = this->TryGetOwner();
	FTransform PawnTransform = FTransform::Identity;

	if (Owner)
	{
		// Get the pawn's rotation to use as reference frame
		PawnTransform = FTransform(Owner->GetActorRotation(), FVector::ZeroVector, FVector::OneVector);
	} else{
		UE_LOG(LogSchola, Warning, TEXT("RayCastSensor is Not Attached to an Actor!"))
	}
	
	// Combine the pawn's transform with the base transform
	// This makes InBaseTransform relative to the pawn's orientation
	FTransform CombinedTransform = InBaseTransform*PawnTransform;
	
	for (int i = 0; i < OutAngles.Num(); i++)
	{
		// Apply the combined transform that includes the pawn's orientation
		OutAngles[i] = CombinedTransform.TransformVector(OutAngles[i]);
		OutAngles[i] = OutAngles[i] + InStart;
		OutAngles[i] = OutAngles[i] + InEndOffset;
	}

	return OutAngles;
}

void URayCastObserver::AppendEmptyTags(FBoxPoint& OutObservations)
{
	for (FName& TrackedTag : TrackedTags)
	{
		OutObservations.Values.Emplace(0.0);
	}
}

void URayCastObserver::HandleRayMiss(FBoxPoint& OutObservations, FVector& Start, FVector& RayEndpoint)
{
	AppendEmptyTags(OutObservations);
	// Tack on the did I hit and hit distance variables
	OutObservations.Values.Emplace(0.0);
	OutObservations.Values.Emplace(0.0);
	if (bDrawDebugLines)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: RaySensor Debug Lines");
		DrawDebugLine(
			GetWorld(),
			Start,
			RayEndpoint,
			DebugMissColor,
			false,
			0,
			0,
			kLineGirth);
	}
}

void URayCastObserver::HandleRayHit(FHitResult& InHitResult, FBoxPoint& OutObservations, FVector& InStart)
{
	// A Precondition is that the HitResult Suceeded, thus Hit.GetActor is always valid

	AActor* HitObject = InHitResult.GetActor();

	TArray<FName>& AttachedTags = HitObject->Tags;

	// Fast Path where we just slap a bunch of zeroes and call it a day
	if (AttachedTags.Num() == 0)
	{
		AppendEmptyTags(OutObservations);
	}
	else
	{
		for (FName& TrackedTag : TrackedTags)
		{
			bool bIsTrackedTagFound = false;
			for (int i = 0; i < AttachedTags.Num() && !bIsTrackedTagFound; i++)
			{
				bIsTrackedTagFound = AttachedTags[i] == TrackedTag;
			}

			OutObservations.Values.Emplace(static_cast<float>(bIsTrackedTagFound));
		}
	}

	if (bDrawDebugLines)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: RaySensor Debug Lines");
		DrawDebugLine(
			GetWorld(),
			InStart,
			InHitResult.ImpactPoint,
			DebugHitColor,
			false,
			0,
			0,
			kLineGirth);

		DrawDebugSphere(
			GetWorld(),
			InHitResult.ImpactPoint - InHitResult.ImpactNormal * kSphereRadius,
			kSphereRadius,
			12,
			DebugHitColor);
	}

	// At this point we have a hit and it has some tags
	// Loop through each tracked tag and check if we find some
	// Always adds 0/1 to the OutObservations for each tag
	OutObservations.Values.Emplace(1.0f);
	OutObservations.Values.Emplace(InHitResult.Time);
}

void URayCastObserver::CollectObservations(FBoxPoint& OutObservations)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: RaySensor Observation Collection");

	AActor* Owner = this->TryGetOwner();
	if (Owner)
	{
		FVector ActorLocation = Owner->GetActorLocation();
		FVector ForwardVector = Owner->GetActorForwardVector();
		FVector Start = (RayStartTransform*FTransform(Owner->GetActorRotation(), FVector::ZeroVector, FVector::OneVector)).GetTranslation() + ActorLocation;
		UE_LOG(LogSchola, Verbose, TEXT(" Actor Location: %s"), *ActorLocation.ToString());
		UE_LOG(LogSchola, Verbose, TEXT(" Raycast starting from: %s"), *Start.ToString());
		TArray<FVector> Endpoints = GenerateRayEndpoints(NumRays, RayDegrees, ForwardVector * RayLength, Start, RayStartTransform, RayEndOffset);

		FCollisionQueryParams TraceParams = FCollisionQueryParams(FName(*FString("RayCastSensor")), this->bTraceComplex, Owner);
		for (auto RayEndpoint : Endpoints)
		{
			FHitResult Hit;
			bool	   bHasHit = GetWorld()->LineTraceSingleByChannel(
				  Hit,
				  Start,
				  RayEndpoint,
				  this->CollisionChannel,
				  TraceParams,
				  FCollisionResponseParams::DefaultResponseParam);

			if (bHasHit)
			{
				HandleRayHit(Hit, OutObservations, Start);
			}
			else
			{
				HandleRayMiss(OutObservations, Start, RayEndpoint);
			}
		}
	}
	else
	{
		UE_LOG(LogSchola, Warning, TEXT("RayCastSensor is Not Attached to an Actor!"))
	}
}

FString URayCastObserver::GenerateId() const
{
	FString Output = FString("Ray");
	// Add the number of rays
	Output.Appendf(TEXT("_Num_%d"), NumRays);
	// Add the angle of the rays
	Output.Appendf(TEXT("_Deg_%.2f"), RayDegrees);
	//Add the Max distance
	Output.Appendf(TEXT("_Max_%.2f"), RayLength);
	Output.Append("_").Append(UEnum::GetValueAsString<ECollisionChannel>(CollisionChannel));
	// Add the tags
	if (TrackedTags.Num() > 0)
	{
		Output.Append("_Tags");
		for (const FName& Tag : TrackedTags)
		{
			Output.Appendf(TEXT("_%s"), *Tag.ToString());
		}
	}
	return Output;
}

#if WITH_EDITOR
void URayCastObserver::DrawDebugLines()
{
	FlushPersistentDebugLines(GetWorld());

	// TODO
	AActor* Owner = nullptr; // = GetOwner();

	if (Owner)
	{
		FVector			ActorLocation = Owner->GetActorLocation();
		FVector			ForwardVector = Owner->GetActorForwardVector();
		FVector			Start =  (RayStartTransform*FTransform(Owner->GetActorRotation(), FVector::ZeroVector, FVector::OneVector)).GetTranslation() + ActorLocation;
		TArray<FVector> Endpoints = GenerateRayEndpoints(NumRays, RayDegrees, ForwardVector * RayLength, Start, RayStartTransform, RayEndOffset);

		for (auto RayEndpoint : Endpoints)
		{
			float DirectionLengthAndSphereRadius = (RayEndpoint - Start).Length();
			float ScaledDirectionLengthAndSphereRadius = RayLength > 0
				? kSphereRadius * DirectionLengthAndSphereRadius / RayLength
				: kSphereRadius;

			DrawDebugLine(
				GetWorld(),
				Start,
				RayEndpoint,
				FColor::MakeRandomColor(),
				true,
				0,
				0,
				kLineGirth);
		}
	}
}

void URayCastObserver::ToggleDebugLines()
{
	if (bDebugLinesEnabled)
	{
		FlushPersistentDebugLines(GetWorld());
	}
	else
	{
		this->DrawDebugLines();
	}

	bDebugLinesEnabled = !bDebugLinesEnabled;
}
#endif