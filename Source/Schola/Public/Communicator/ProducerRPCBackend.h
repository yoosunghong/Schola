// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "./CallData.h"
#include "./AbstractRPCBackend.h"
#include "./ComBackendInterface.h"
#include "Common/CommonInterfaces.h"

// A debug int for tracking logs across Exchanges
static int ProducerID = 0;

template <class ServiceType, typename RequestType, typename ResponseType>
class ProducerRPCWorker : public FRunnable
{
private:
	typedef CallData<ServiceType, RequestType, ResponseType> _CallData;
	// CQueue owned by this workers parent
	ServerCompletionQueue* CQueue;
	// The ID of this worker
	int LocalID;

public:
	// The thread object this worker runs on
	FRunnableThread* Thread = nullptr;

	ProducerRPCWorker(ServerCompletionQueue* CQueue, int ID)
	{
		LocalID = ID;
		this->CQueue = CQueue;
	}

	~ProducerRPCWorker()
	{
		delete Thread;
	}
	/**
	 * @brief Initialize the worker
	 * @return True. Since initialization cannot fail
	 */
	virtual bool Init()
	{
		// Do Nothing here
		return true;
	}

	/**
	 * @brief This workers main method
	 * @return A status code representing the result
	 */
	virtual uint32 Run()
	{
		UE_LOG(LogScholaCommunicator, Verbose, TEXT("Producer Thread %d Started"), LocalID);
		void* tag = nullptr; // uniquely identifies a request.
		bool  ok = true;
		while (true)
		{
			UE_LOG(LogScholaCommunicator, VeryVerbose, TEXT("Waiting for Event on Producer queue %d"), LocalID);
			bool Status = CQueue->Next(&tag, &ok);
			// Gotta check this way because if the queue was empty we also get a nonsense tag
			if (!Status)
			{
				// Queue drained so we can exit
				UE_LOG(LogScholaCommunicator, Verbose, TEXT("Producer Queue %d Drained and Shutdown"), LocalID);
				return -1;
			}
			else if (!ok)
			{
				// we can assume this since other events will have a tag
				// if we get nullptr and !ok then the Queue must be empty and therefore Status=False
				UE_LOG(LogScholaCommunicator, Warning, TEXT("Invalid Event in Producer %d Completion Queue"), LocalID);
				// This tag was cleanupable so clean it up
				if (tag != nullptr)
				{
					_CallData* CallData = static_cast<_CallData*>(tag);
					CallData->CleanUp();
				}
			}
			else
			{
				_CallData* CallData = static_cast<_CallData*>(tag);
				CallData->DoWork();
			}
		}
	}

	/**
	 * @brief Start the worker
	 */
	void Start()
	{
		UE_LOG(LogScholaCommunicator, Verbose, TEXT("Starting Producer Worker %d"), LocalID);
		Thread = FRunnableThread::Create(this, TEXT("ProducerRPCWorker"), 0, TPri_Normal);
	}

	/**
	 * @brief Shutdown the worker and it's associated completion queue
	 */
	virtual void Stop()
	{
		UE_LOG(LogScholaCommunicator, Verbose, TEXT("Shutting Down Producer Queue %d"), LocalID);
		CQueue->Shutdown();
		// Wait for the CQueue to drain
		if (Thread != nullptr)
		{
			Thread->WaitForCompletion();
		}
	}

	/**
	 * @brief Unused. Called when the thread completes
	 */
	virtual void Exit()
	{
		// Called on Completion so do nothing
	}
};

template <class ServiceType, typename RequestType, typename ResponseType>
class ProducerRPCBackend : public RPCBackend<ServiceType, RequestType, ResponseType>, public IProducerBackendInterface<ResponseType>
{
private:
	int														 LocalID = 0;
	int														 MsgID = 0;
	typedef CallData<ServiceType, RequestType, ResponseType> _CallData;
	// Note these are inverted since we are sending response, before the request arrives from gRPC perspective
	ProducerRPCWorker<ServiceType, RequestType, ResponseType>* Worker;
	using RPCBackend = RPCBackend<ServiceType, RequestType, ResponseType>;

public:
	ProducerRPCBackend(RPCBackend::AsyncRPCHandle TargetRPC, std::shared_ptr<ServiceType> Service, std::unique_ptr<ServerCompletionQueue> CQueue)
		: RPCBackend(TargetRPC, Service, std::move(CQueue))
	{
		LocalID = ExchangeID++;
		this->Worker = new ProducerRPCWorker<ServiceType, RequestType, ResponseType>(RPCBackend::_CQueue.get(), LocalID);
	}

	~ProducerRPCBackend()
	{
		UE_LOG(LogScholaCommunicator, Warning, TEXT("Manually Deleting ProducerRPC Backend %d"), LocalID);
		Shutdown();
		delete this->Worker;
	}

	void SendProtobufMessage(ResponseType* Response) override
	{
		_CallData* CallDataPtr = new _CallData(this->Service.get(), this->_CQueue.get(), this->TargetRPC, false, false);
		CallDataPtr->SetResponse(Response);
		int TempID = MsgID++;
		// UE_LOG(LogScholaCommunicator, VeryVerbose, TEXT("Adding Message %d to producer queue %d \n %s"), TempID, LocalID, *FString(Response->DebugString().c_str()));
		CallDataPtr->Id = TempID;
		CallDataPtr->Create();
	}

	virtual void Initialize() {};

	virtual void Start()
	{
		Worker->Start();
	}

	virtual void Establish() {};

	virtual void Shutdown() override
	{
		this->Worker->Stop();
	};

	virtual void Restart() {};
};