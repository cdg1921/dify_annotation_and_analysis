from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import uuid4

from configs import dify_config
from core.file import File
from core.variables.exc import VariableError
from core.variables.segments import (
    ArrayAnySegment,
    ArrayFileSegment,
    ArrayNumberSegment,
    ArrayObjectSegment,
    ArraySegment,
    ArrayStringSegment,
    FileSegment,
    FloatSegment,
    IntegerSegment,
    NoneSegment,
    ObjectSegment,
    Segment,
    StringSegment,
)
from core.variables.types import SegmentType
from core.variables.variables import (
    ArrayAnyVariable,
    ArrayFileVariable,
    ArrayNumberVariable,
    ArrayObjectVariable,
    ArrayStringVariable,
    FileVariable,
    FloatVariable,
    IntegerVariable,
    NoneVariable,
    ObjectVariable,
    SecretVariable,
    StringVariable,
    Variable,
)
from core.workflow.constants import CONVERSATION_VARIABLE_NODE_ID, ENVIRONMENT_VARIABLE_NODE_ID


class InvalidSelectorError(ValueError):
    pass


class UnsupportedSegmentTypeError(Exception):
    pass

# cdg: 定义变量类型到变量类的映射
# Define the constant
SEGMENT_TO_VARIABLE_MAP = {
    StringSegment: StringVariable,
    IntegerSegment: IntegerVariable,
    FloatSegment: FloatVariable,
    ObjectSegment: ObjectVariable,
    FileSegment: FileVariable,
    ArrayStringSegment: ArrayStringVariable,
    ArrayNumberSegment: ArrayNumberVariable,
    ArrayObjectSegment: ArrayObjectVariable,
    ArrayFileSegment: ArrayFileVariable,
    ArrayAnySegment: ArrayAnyVariable,
    NoneSegment: NoneVariable,
}

# cdg: 从映射构建对话变量
def build_conversation_variable_from_mapping(mapping: Mapping[str, Any], /) -> Variable:
    if not mapping.get("name"): # cdg: 如果变量名不存在，则抛出异常
        raise VariableError("missing name")
    # cdg: 从映射构建对话变量
    return _build_variable_from_mapping(mapping=mapping, selector=[CONVERSATION_VARIABLE_NODE_ID, mapping["name"]])

# cdg: 从映射构建环境变量
def build_environment_variable_from_mapping(mapping: Mapping[str, Any], /) -> Variable:
    if not mapping.get("name"): # cdg: 如果变量名不存在，则抛出异常
        raise VariableError("missing name")
    # cdg: 从映射构建环境变量
    return _build_variable_from_mapping(mapping=mapping, selector=[ENVIRONMENT_VARIABLE_NODE_ID, mapping["name"]])

# cdg: 从映射构建变量
def _build_variable_from_mapping(*, mapping: Mapping[str, Any], selector: Sequence[str]) -> Variable:
    """
    This factory function is used to create the environment variable or the conversation variable,
    not support the File type.
    """
    if (value_type := mapping.get("value_type")) is None:
        raise VariableError("missing value type")
    if (value := mapping.get("value")) is None:
        raise VariableError("missing value")
    # FIXME: using Any here, fix it later # cdg: 使用Any类型，待修复
    result: Any
    match value_type:
        case SegmentType.STRING:
            result = StringVariable.model_validate(mapping)
        case SegmentType.SECR#ET:
            result = SecretVariable.model_validate(mapping)
        case SegmentType.NUMBER if isinstance(value, int):
            result = IntegerVariable.model_validate(mapping)
        case SegmentType.NUMBER if isinstance(value, float):
            result = FloatVariable.model_validate(mapping)
        case SegmentType.NUMBER if not isinstance(value, float | int):
            raise VariableError(f"invalid number value {value}")
        case SegmentType.OBJECT if isinstance(value, dict):
            result = ObjectVariable.model_validate(mapping)
        case SegmentType.ARRAY_STRING if isinstance(value, list):
            result = ArrayStringVariable.model_validate(mapping)
        case SegmentType.ARRAY_NUMBER if isinstance(value, list):
            result = ArrayNumberVariable.model_validate(mapping)
        case SegmentType.ARRAY_OBJECT if isinstance(value, list):
            result = ArrayObjectVariable.model_validate(mapping)
        case _:
            raise VariableError(f"not supported value type {value_type}")
    if result.size > dify_config.MAX_VARIABLE_SIZE:
        raise VariableError(f"variable size {result.size} exceeds limit {dify_config.MAX_VARIABLE_SIZE}")
    if not result.selector:
        result = result.model_copy(update={"selector": selector})
    return cast(Variable, result)

# cdg: 构建变量类型对象（一个类型对象Segment包括value_type和value两个属性，value_type表示类型，value表示值）
def build_segment(value: Any, /) -> Segment:
    if value is None: # cdg: 如果值为空，则返回空段
        return NoneSegment()
    if isinstance(value, str): # cdg: 如果值为字符串，则返回字符串对象  
        return StringSegment(value=value)
    if isinstance(value, int): # cdg: 如果值为整数，则返回整数对象
        return IntegerSegment(value=value)
    if isinstance(value, float): # cdg: 如果值为浮点数，则返回浮点数对象
        return FloatSegment(value=value)
    if isinstance(value, dict): # cdg: 如果值为字典，则返回对象对象
        return ObjectSegment(value=value)
    if isinstance(value, File): # cdg: 如果值为文件，则返回文件对象
        return FileSegment(value=value)
    if isinstance(value, list): # cdg: 如果值为列表，则返回数组对象
        items = [build_segment(item) for item in value] # cdg: 构建列表中的每个元素
        types = {item.value_type for item in items} # cdg: 获取列表中每个元素的类型
        if len(types) != 1 or all(isinstance(item, ArraySegment) for item in items): # cdg: 如果列表中每个元素的类型不唯一，或者所有元素都是数组，则返回数组对象
            return ArrayAnySegment(value=value)
        match types.pop(): # cdg: 根据列表中每个元素的类型，返回不同的数组对象
            case SegmentType.STRING: # cdg: 如果列表中每个元素的类型为字符串，则返回字符串数组对象
                return ArrayStringSegment(value=value)
            case SegmentType.NUMBER: # cdg: 如果列表中每个元素的类型为整数，则返回整数数组对象
                return ArrayNumberSegment(value=value)
            case SegmentType.OBJECT: # cdg: 如果列表中每个元素的类型为对象，则返回对象数组对象
                return ArrayObjectSegment(value=value)
            case SegmentType.FILE: # cdg: 如果列表中每个元素的类型为文件，则返回文件数组对象
                return ArrayFileSegment(value=value)
            case SegmentType.NONE: # cdg: 如果列表中每个元素的类型为空，则返回空数组对象
                return ArrayAnySegment(value=value)
            case _: # cdg: 如果列表中每个元素的类型为其他类型，则抛出异常
                raise ValueError(f"not supported value {value}")
    raise ValueError(f"not supported value {value}")

# cdg: 将类型对象转换为变量
def segment_to_variable(
    *,
    segment: Segment,
    selector: Sequence[str],
    id: str | None = None,
    name: str | None = None,
    description: str = "",
) -> Variable:
    if isinstance(segment, Variable):
        return segment
    name = name or selector[-1]
    id = id or str(uuid4())

    segment_type = type(segment)
    if segment_type not in SEGMENT_TO_VARIABLE_MAP:
        raise UnsupportedSegmentTypeError(f"not supported segment type {segment_type}")

    variable_class = SEGMENT_TO_VARIABLE_MAP[segment_type]
    return cast(
        Variable,
        variable_class(
            id=id,
            name=name,
            description=description,
            value=segment.value,
            selector=selector,
        ),
    )
