// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "./CallData.h"
#include "./AbstractRPCBackend.h"
#include "./ComBackendInterface.h"
#include "Common/CommonInterfaces.h"

template <class ServiceType, typename RequestType, typename ResponseType>
class PollingRPCWorker : public FRunnable
{
private:
	typedef CallData<ServiceType, RequestType, ResponseType> _PollCallData;
	// CQueue owned by this workers parent
	ServerCompletionQueue* CQueue;

public:
	FRunnableThread*	Thread = nullptr;
	TQueue<RequestType> Requests = TQueue<RequestType>();

	PollingRPCWorker(ServerCompletionQueue* CQueue)
	{
		this->CQueue = CQueue;
	}

	~PollingRPCWorker()
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
		UE_LOG(LogScholaCommunicator, Verbose, TEXT("Polling Thread Started"));
		// This thread will loop through and fulfill promises etc on the exchange server
		void* tag = nullptr; // uniquely identifies a request.
		bool  ok = true;
		while (true)
		{
			// wait until some message is ready
			UE_LOG(LogScholaCommunicator, VeryVerbose, TEXT("Waiting for Event on Polling queue"));
			bool Status = CQueue->Next(&tag, &ok);
			if (!Status)
			{
				// Queue drained so we can exit
				UE_LOG(LogScholaCommunicator, Warning, TEXT("Polling Queue Drained and Shutdown"));
				return -1;
			}
			else if (!ok)
			{
				UE_LOG(LogScholaCommunicator, Warning, TEXT("Invalid Event in Polling Completion Queue"));
				// Clean up the message
				if (tag != nullptr)
				{
					_PollCallData* CallData = static_cast<_PollCallData*>(tag);
					CallData->CleanUp();
				}
				UE_LOG(LogScholaCommunicator, Warning, TEXT("Returning From Completion Queue"));
			}
			else
			{
				// Normal event handling
				UE_LOG(LogScholaCommunicator, VeryVerbose, TEXT("Queue had an Event!"));
				_PollCallData* CallData = static_cast<_PollCallData*>(tag);
				
				if (CallData->IsReady())
				{
					UE_LOG(LogScholaCommunicator, VeryVerbose, TEXT("Message Received on Poll!"));
					Requests.Enqueue(CallData->GetRequest());
				}
				CallData->DoWork();
			}
		}
	}

	/**
	 * @brief Start the worker
	 */
	void Start()
	{
		UE_LOG(LogScholaCommunicator, Verbose, TEXT("Starting Polling Worker"));
		Thread = FRunnableThread::Create(this, TEXT("PollingRPCWorker"), 0, TPri_Normal);
	}

	/**
	 * @brief Shutdown the worker and it's associated completion queue
	 */
	virtual void Stop()
	{
		UE_LOG(LogScholaCommunicator, Warning, TEXT("Shutting Down Polling Queue"));
		CQueue->Shutdown();
		// Wait for the CQueue to Drain
		if (Thread != nullptr)
		{
			Thread->WaitForCompletion();
		}
	}

	virtual void Exit()
	{
		// Called on Completion so do nothing
	}
};

template <class ServiceType, typename RequestType, typename ResponseType>
class PollingRPCBackend : public RPCBackend<ServiceType, RequestType, ResponseType>, public IPollingBackendInterface<RequestType>
{
private:
	using _PollCallData = CallData<ServiceType, RequestType, ResponseType>;
	PollingRPCWorker<ServiceType, RequestType, ResponseType>* Worker;
	using RPCBackend = RPCBackend<ServiceType, RequestType, ResponseType>;

public:
	PollingRPCBackend(RPCBackend::AsyncRPCHandle TargetRPC, std::shared_ptr<ServiceType> Service, std::unique_ptr<ServerCompletionQueue> CQueue)
		: RPCBackend(TargetRPC, Service, std::move(CQueue))
	{
		this->Worker = new PollingRPCWorker<ServiceType, RequestType, ResponseType>(RPCBackend::_CQueue.get());
	}

	~PollingRPCBackend()
	{
		Shutdown();
		delete this->Worker;
	}

	TOptional<const RequestType*> Poll() override
	{
		if (Worker->Requests.IsEmpty())
		{
			return TOptional<const RequestType*>();
		}
		else
		{
			// Deque the front of the message queue
			RequestType RequestRef;
			Worker->Requests.Dequeue(RequestRef);
			return TOptional<const RequestType*>(new RequestType(RequestRef));
		}
	}

	virtual void Initialize(){};

	virtual void Start() override
	{
		new _PollCallData(this->Service.get(), this->_CQueue.get(), RPCBackend::TargetRPC, true);
		this->Worker->Start();
	}
	virtual void Establish(){};

	virtual void Shutdown() override
	{
		this->Worker->Stop();
	};

	virtual void Restart(){};
};