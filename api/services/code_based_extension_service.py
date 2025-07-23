from extensions.ext_code_based_extension import code_based_extension


# cdg: 代码扩展服务，用于获取代码扩展。
class CodeBasedExtensionService:
    # cdg: 获取代码扩展。
    @staticmethod
    def get_code_based_extension(module: str) -> list[dict]:
        module_extensions = code_based_extension.module_extensions(module)

        # cdg: 返回代码扩展列表，过滤掉内置的代码扩展。
        return [
            {
                "name": module_extension.name,
                "label": module_extension.label,
                "form_schema": module_extension.form_schema,
            }
            for module_extension in module_extensions
            if not module_extension.builtin
        ]
