import schola.generated.Definitions_pb2 as _Definitions_pb2
import schola.generated.State_pb2 as _State_pb2
import schola.generated.StateUpdates_pb2 as _StateUpdates_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

CLOSED: CommunicatorStatus
DESCRIPTOR: _descriptor.FileDescriptor
DISABLED: AutoResetType
ERROR: CommunicatorStatus
GOOD: CommunicatorStatus
NEXTSTEP: AutoResetType
SAMESTEP: AutoResetType

class EnvironmentReset(_message.Message):
    __slots__ = ["options", "seed"]
    class OptionsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    options: _containers.ScalarMap[str, str]
    seed: int
    def __init__(self, seed: _Optional[int] = ..., options: _Optional[_Mapping[str, str]] = ...) -> None: ...

class EnvironmentStateUpdate(_message.Message):
    __slots__ = ["reset", "step"]
    RESET_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    reset: EnvironmentReset
    step: _StateUpdates_pb2.EnvironmentStep
    def __init__(self, reset: _Optional[_Union[EnvironmentReset, _Mapping]] = ..., step: _Optional[_Union[_StateUpdates_pb2.EnvironmentStep, _Mapping]] = ...) -> None: ...

class GymConnectorStartRequest(_message.Message):
    __slots__ = ["autoreset_type"]
    AUTORESET_TYPE_FIELD_NUMBER: _ClassVar[int]
    autoreset_type: AutoResetType
    def __init__(self, autoreset_type: _Optional[_Union[AutoResetType, str]] = ...) -> None: ...

class GymConnectorStartResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class InitialTrainingStateRequest(_message.Message):
    __slots__ = ["environment_state_requests"]
    class EnvironmentStateRequestsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: InititalEnvironmentStateRequest
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[InititalEnvironmentStateRequest, _Mapping]] = ...) -> None: ...
    ENVIRONMENT_STATE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    environment_state_requests: _containers.MessageMap[int, InititalEnvironmentStateRequest]
    def __init__(self, environment_state_requests: _Optional[_Mapping[int, InititalEnvironmentStateRequest]] = ...) -> None: ...

class InititalEnvironmentStateRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class TrainingDefinitionRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class TrainingStateUpdate(_message.Message):
    __slots__ = ["status", "updates"]
    class UpdatesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: EnvironmentStateUpdate
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[EnvironmentStateUpdate, _Mapping]] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPDATES_FIELD_NUMBER: _ClassVar[int]
    status: CommunicatorStatus
    updates: _containers.MessageMap[int, EnvironmentStateUpdate]
    def __init__(self, updates: _Optional[_Mapping[int, EnvironmentStateUpdate]] = ..., status: _Optional[_Union[CommunicatorStatus, str]] = ...) -> None: ...

class CommunicatorStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class AutoResetType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
