from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

from core.app.app_config.entities import VariableEntityType
from core.file import File, FileUploadConfig
from factories import file_factory

if TYPE_CHECKING:
    from core.app.app_config.entities import VariableEntity


class BaseAppGenerator:
    def _prepare_user_inputs(
        self,
        *,
        user_inputs: Optional[Mapping[str, Any]],
        variables: Sequence["VariableEntity"],
        tenant_id: str,
    ) -> Mapping[str, Any]:
        # cdg:user_inputs的处理
        user_inputs = user_inputs or {}
        # Filter input variables from form configuration, handle required fields, default values, and option values
        # cdg:对每个变量进行格式检查
        user_inputs = {
            var.variable: self._validate_inputs(value=user_inputs.get(var.variable), variable_entity=var)
            for var in variables
        }
        # cdg:_sanitize_value删除value中的空字符串
        user_inputs = {k: self._sanitize_value(v) for k, v in user_inputs.items()}

        # cdg:文件的处理， VariableEntity实例中VariableEntityType为"file"
        # Convert files in inputs to File
        entity_dictionary = {item.variable: item for item in variables}
        # Convert single file to File
        files_inputs = {
            k: file_factory.build_from_mapping(
                mapping=v,
                tenant_id=tenant_id,
                config=FileUploadConfig(
                    allowed_file_types=entity_dictionary[k].allowed_file_types,
                    allowed_file_extensions=entity_dictionary[k].allowed_file_extensions,
                    allowed_file_upload_methods=entity_dictionary[k].allowed_file_upload_methods,
                ),
            )
            for k, v in user_inputs.items()
            if isinstance(v, dict) and entity_dictionary[k].type == VariableEntityType.FILE
        }

        # cdg:文件列表的处理
        # Convert list of files to File
        file_list_inputs = {
            k: file_factory.build_from_mappings(
                mappings=v,
                tenant_id=tenant_id,
                config=FileUploadConfig(
                    allowed_file_types=entity_dictionary[k].allowed_file_types,
                    allowed_file_extensions=entity_dictionary[k].allowed_file_extensions,
                    allowed_file_upload_methods=entity_dictionary[k].allowed_file_upload_methods,
                ),
            )
            for k, v in user_inputs.items()
            if isinstance(v, list)
            # Ensure skip List<File>
            and all(isinstance(item, dict) for item in v)
            and entity_dictionary[k].type == VariableEntityType.FILE_LIST
        }

        # cdg:user_inputs、files_inputs、file_list_inputs合并成一个user_inputs字典
        # Merge all inputs
        user_inputs = {**user_inputs, **files_inputs, **file_list_inputs}

        # cdg:检查是否所有files都转为File
        # Check if all files are converted to File
        if any(filter(lambda v: isinstance(v, dict), user_inputs.values())):
            raise ValueError("Invalid input type")
        if any(
            filter(lambda v: isinstance(v, dict), filter(lambda item: isinstance(item, list), user_inputs.values()))
        ):
            raise ValueError("Invalid input type")

        return user_inputs

    def _validate_inputs(
        self,
        *,
        variable_entity: "VariableEntity",
        value: Any,
    ):
        # cdg:对于必填参数，取值不能为空
        if value is None:
            if variable_entity.required:
                raise ValueError(f"{variable_entity.variable} is required in input form")
            return value

        # cdg:TEXT_INPUT、SELECT、PARAGRAPH等类型的变量，取值类型必须为str
        if variable_entity.type in {
            VariableEntityType.TEXT_INPUT,
            VariableEntityType.SELECT,
            VariableEntityType.PARAGRAPH,
        } and not isinstance(value, str):
            raise ValueError(
                f"(type '{variable_entity.type}') {variable_entity.variable} in input form must be a string"
            )

        # cdg:数值类型数据检查
        if variable_entity.type == VariableEntityType.NUMBER and isinstance(value, str):
            # handle empty string case
            if not value.strip():
                return None
            # may raise ValueError if user_input_value is not a valid number
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                raise ValueError(f"{variable_entity.variable} in input form must be a valid number")

        match variable_entity.type:
            case VariableEntityType.SELECT:
                # cdg:可选项参数值不在给定选项中，则报错
                if value not in variable_entity.options:
                    raise ValueError(
                        f"{variable_entity.variable} in input form must be one of the following: "
                        f"{variable_entity.options}"
                    )
            case VariableEntityType.TEXT_INPUT | VariableEntityType.PARAGRAPH:
                # cdg:文本类型长度检查
                if variable_entity.max_length and len(value) > variable_entity.max_length:
                    raise ValueError(
                        f"{variable_entity.variable} in input form must be less than {variable_entity.max_length} "
                        "characters"
                    )
            case VariableEntityType.FILE:
                # cdg:文件类型参数检查
                if not isinstance(value, dict) and not isinstance(value, File):
                    raise ValueError(f"{variable_entity.variable} in input form must be a file")
            case VariableEntityType.FILE_LIST:
                # if number of files exceeds the limit, raise ValueError
                if not (
                    isinstance(value, list)
                    and (all(isinstance(item, dict) for item in value) or all(isinstance(item, File) for item in value))
                ):
                    # cdg:FILE_LIST类型参数检查
                    raise ValueError(f"{variable_entity.variable} in input form must be a list of files")

                if variable_entity.max_length and len(value) > variable_entity.max_length:
                    raise ValueError(
                        f"{variable_entity.variable} in input form must be less than {variable_entity.max_length} files"
                    )

        return value

    def _sanitize_value(self, value: Any) -> Any:
        # cdg:删除空字符串
        if isinstance(value, str):
            return value.replace("\x00", "")
        return value
