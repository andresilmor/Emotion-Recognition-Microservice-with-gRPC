from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EmotionRecognitionInferenceReply(_message.Message):
    __slots__ = ["categorical", "continuous"]
    class ContinuousEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    CATEGORICAL_FIELD_NUMBER: _ClassVar[int]
    CONTINUOUS_FIELD_NUMBER: _ClassVar[int]
    categorical: _containers.RepeatedScalarFieldContainer[str]
    continuous: _containers.ScalarMap[str, float]
    def __init__(self, continuous: _Optional[_Mapping[str, float]] = ..., categorical: _Optional[_Iterable[str]] = ...) -> None: ...

class EmotionRecognitionRequest(_message.Message):
    __slots__ = ["image", "personBox"]
    class PersonBoxEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    PERSONBOX_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    personBox: _containers.ScalarMap[str, int]
    def __init__(self, image: _Optional[bytes] = ..., personBox: _Optional[_Mapping[str, int]] = ...) -> None: ...
