from typing import Union, cast

from core.model_runtime.entities.provider_entities import CredentialFormSchema, FormType

# cdg:通用验证器，供应商验证和模型验证都会用到它：[ModelCredentialSchemaValidator, ProviderCredentialSchemaValidator] -> CommonValidator
class CommonValidator:
    def _validate_and_filter_credential_form_schemas(
        self, credential_form_schemas: list[CredentialFormSchema], credentials: dict
    ) -> dict:
        # cdg:credential_form_schemas，预定义credential，作为标准；credentials，待验证的credential
        need_validate_credential_form_schema_map = {}
        # cdg:每一个credential_form_schema包含一个变量的结构体，对应一个变量；credential_form_schemas是变量结构体列表
        for credential_form_schema in credential_form_schemas:
            # cdg:不是显式变量，直接作为验证对象；显式变量，需要验证变量名和值
            if not credential_form_schema.show_on:
                need_validate_credential_form_schema_map[credential_form_schema.variable] = credential_form_schema
                continue

            all_show_on_match = True
            for show_on_object in credential_form_schema.show_on:
                # cdg:变量名验证
                if show_on_object.variable not in credentials:
                    all_show_on_match = False
                    break

                # cdg:变量值验证
                if credentials[show_on_object.variable] != show_on_object.value:
                    all_show_on_match = False
                    break

            # cdg:名称和值都匹配成功，通过验证
            if all_show_on_match:
                need_validate_credential_form_schema_map[credential_form_schema.variable] = credential_form_schema

        # cdg:对于每一个需要验证的CredentialFormSchema
        # Iterate over the remaining credential_form_schemas, verify each credential_form_schema
        validated_credentials = {}
        for credential_form_schema in need_validate_credential_form_schema_map.values():
            # add the value of the credential_form_schema corresponding to it to validated_credentials
            result = self._validate_credential_form_schema(credential_form_schema, credentials)
            if result:
                validated_credentials[credential_form_schema.variable] = result

        return validated_credentials

    def _validate_credential_form_schema(
        self, credential_form_schema: CredentialFormSchema, credentials: dict
    ) -> Union[str, bool, None]:
        """
        Validate credential form schema

        :param credential_form_schema: credential form schema
        :param credentials: credentials
        :return: validated credential form schema value
        """
        #  If the variable does not exist in credentials
        #
        value: Union[str, bool, None] = None   # cdg:标识value的取值类型，起提示说明的作用
        # cdg:credential_form_schema的变量是应有的变量，credentials的变量是用户提供的变量；
        # 如果用户没提供或者用户提供的变量值为None，首先检查该变量是否为必填（required）；如果是必填，则报错；如果非必填，则取默认值
        if credential_form_schema.variable not in credentials or not credentials[credential_form_schema.variable]:
            # If required is True, an exception is thrown
            if credential_form_schema.required:
                raise ValueError(f"Variable {credential_form_schema.variable} is required")
            else:
                # Get the value of default
                if credential_form_schema.default:
                    # If it exists, add it to validated_credentials
                    return credential_form_schema.default
                else:
                    # If default does not exist, skip
                    return None
        # cdg:将每个变量值转为str类型，正常变量名称都应该是str类型，如果不是str类型，后续当str使用时会报错
        # Get the value corresponding to the variable from credentials
        value = cast(str, credentials[credential_form_schema.variable])

        # cdg:max_length=0，取值可能不是字符串类型，不做长度检查
        # If max_length=0, no validation is performed
        if credential_form_schema.max_length:
            # cdg:如果取值产出指定长度范围，则报错
            if len(value) > credential_form_schema.max_length:
                raise ValueError(
                    f"Variable {credential_form_schema.variable} length should not"
                    f" greater than {credential_form_schema.max_length}"
                )

        # check the type of value
        if not isinstance(value, str):
            raise ValueError(f"Variable {credential_form_schema.variable} should be string")

        # cdg:单选或多选，则检查value是否在为给定的选项中
        if credential_form_schema.type in {FormType.SELECT, FormType.RADIO}:
            # If the value is in options, no validation is performed
            if credential_form_schema.options:
                if value not in [option.value for option in credential_form_schema.options]:
                    raise ValueError(f"Variable {credential_form_schema.variable} is not in options")

        # cdg:如果是SWITCH类型，则只能取值"true"或"false"
        if credential_form_schema.type == FormType.SWITCH:
            # If the value is not in ['true', 'false'], an exception is thrown
            if value.lower() not in {"true", "false"}:
                raise ValueError(f"Variable {credential_form_schema.variable} should be true or false")

            value = value.lower() == "true"

        # cdg:返回验证通过的值（字符串类型）
        return value
