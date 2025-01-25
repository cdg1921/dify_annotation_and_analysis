from core.model_runtime.entities.provider_entities import ProviderCredentialSchema
from core.model_runtime.schema_validators.common_validator import CommonValidator

# cdg:这么多行代码，实际是根据给定的provider_credential_schema调用CommonValidator._validate_and_filter_credential_form_schemas
class ProviderCredentialSchemaValidator(CommonValidator):
    def __init__(self, provider_credential_schema: ProviderCredentialSchema):
        self.provider_credential_schema = provider_credential_schema

    def validate_and_filter(self, credentials: dict) -> dict:
        """
        Validate provider credentials

        :param credentials: provider credentials
        :return: validated provider credentials
        """
        # get the credential_form_schemas in provider_credential_schema
        credential_form_schemas = self.provider_credential_schema.credential_form_schemas

        return self._validate_and_filter_credential_form_schemas(credential_form_schemas, credentials)
