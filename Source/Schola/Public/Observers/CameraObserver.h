// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/LogSchola.h"
#include "TextureResource.h"
#include "Engine/Texture2D.h"  
#include "RenderUtils.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Observers/AbstractObservers.h"
#include "CameraObserver.generated.h"

/**
 * @brief An observer that collects 2D observations from a camera in the environment.
 * 
 * @details sensor uses a SceneCaptureComponent2D (https://dev.epicgames.com/documentation/en-us/unreal-engine/1.7---scene-capture-2d?application_version=4.27) and a RenderTarget(https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/RenderTarget?application_version=5.5)
 * to capture images from the environment. To use this sensor, a SceneCaptureComponent2D and a RenderTarget must be first created in the Unreal Editor, and selected in the sensor settings.
 * 
 * Change the CompositeMode setting in the SceneCaptureComponent2D to choose the mode of operation of the camera (RGB, RGB-depth, depth-only).
 * 
 * RenderTarget setting recommendations:
 *   Advanced --> Shared  = true
 * 
 * SceneCaptureComponent2D setting recommendations:
 *	 bCaptureEveryFrame = true
 *	 bRenderInMainRenderer = true
 *   CompositeMode = overwrite
 */
UCLASS(Blueprintable)
class SCHOLA_API UCameraObserver : public UBoxObserver
{
	GENERATED_BODY()

protected:
	/** The width of the captured image. */
	UPROPERTY(BlueprintReadOnly, Category = "Sensor Properties")
	int Width;

	/** The height of the captured image. */
	UPROPERTY(BlueprintReadOnly, Category = "Sensor Properties")
	int Height;

public:

	/** The Camera Component to use for capturing images. */
	UPROPERTY(EditAnywhere, meta = (UseComponentPicker, AllowAnyActor, AllowedClasses = "SceneCaptureComponent2D"), Category = "Sensor Properties")
	FComponentReference SceneCaptureCompRef;

	/** The Render Target where the captured images are stored to. */
	UPROPERTY(EditAnywhere, Category = "Sensor Properties")
	TObjectPtr<UTextureRenderTarget2D> RenderTarget;

	/** Whether the R channel is observed. */
	UPROPERTY(EditAnywhere, Category = "Sensor Properties")
	bool bObserveChannelR = true;

	/** Whether the G channel is observed. */
	UPROPERTY(EditAnywhere, Category = "Sensor Properties")
	bool bObserveChannelG = false;

	/** Whether the B channelis observed. */
	UPROPERTY(EditAnywhere, Category = "Sensor Properties")
	bool bObserveChannelB = false;

	/** Whether the A channel is observed. */
	UPROPERTY(EditAnywhere, Category = "Sensor Properties")
	bool bObserveChannelA = false;

	FBoxSpace GetObservationSpace() const;
	
	/** Determines whether the specified channel is used. */
	bool IsChannelUsed(USceneCaptureComponent2D* CapComponent, FName ChannelName) const;

	/** Set bObserveChannelR, bObserveChannelG, bObserveChannelB, bObserveChannelA according to the camera mode. */
	void UpdateChannelBooleans();

	FString GenerateId() const override;

# if WITH_EDITOR
	/** Set the editibility of bObserveChannelR, bObserveChannelG, bObserveChannelB, and bObserveChannelA according to the camera mode.  */
	virtual bool CanEditChange(const FProperty* InProperty) const override;
# endif

	/**
	 * @brief Collect observations about the environment state.
	 * @param[out] OutObservations A BoxPoint that will be updated with the outputs of this sensor.
	 */
	virtual void CollectObservations(FBoxPoint& OutObservations) override;
	
	void InitializeObserver() ;
};
