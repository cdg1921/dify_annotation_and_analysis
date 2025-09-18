from typing import Any

import flask_restful  # type: ignore
from flask_login import current_user  # type: ignore
from flask_restful import Resource, fields, marshal_with
from werkzeug.exceptions import Forbidden

from extensions.ext_database import db
from libs.helper import TimestampField
from libs.login import login_required
from models.dataset import Dataset
from models.model import ApiToken, App

from . import api
from .wraps import account_initialization_required, setup_required

# cdg:API密钥字段，常与marshal_with装饰器一起使用，保证返回的API密钥字段符合预期。
api_key_fields = {
    "id": fields.String,
    "type": fields.String,
    "token": fields.String,
    "last_used_at": TimestampField,
    "created_at": TimestampField,
}

# cdg:API密钥列表字段
api_key_list = {"data": fields.List(fields.Nested(api_key_fields), attribute="items")}


def _get_resource(resource_id, tenant_id, resource_model):
    resource = resource_model.query.filter_by(id=resource_id, tenant_id=tenant_id).first()

    if resource is None:
        flask_restful.abort(404, message=f"{resource_model.__name__} not found.")

    return resource

# cdg:获取API密钥列表资源
class BaseApiKeyListResource(Resource):
    method_decorators = [account_initialization_required, login_required, setup_required]

    resource_type: str | None = None
    resource_model: Any = None
    resource_id_field: str | None = None
    token_prefix: str | None = None
    max_keys = 10

    # cdg:获取API密钥列表，通过marshal_with装饰器，保证返回的API密钥列表字段与api_key_list字段一致。
    @marshal_with(api_key_list)
    def get(self, resource_id):
        assert self.resource_id_field is not None, "resource_id_field must be set"
        resource_id = str(resource_id)
        _get_resource(resource_id, current_user.current_tenant_id, self.resource_model)
        keys = (
            db.session.query(ApiToken)
            .filter(ApiToken.type == self.resource_type, getattr(ApiToken, self.resource_id_field) == resource_id)
            .all()
        )
        return {"items": keys}

    # cdg:创建API密钥，通过marshal_with装饰器，保证返回的API密钥字段与api_key_fields字段一致。
    @marshal_with(api_key_fields)
    def post(self, resource_id):
        assert self.resource_id_field is not None, "resource_id_field must be set"
        resource_id = str(resource_id)
        _get_resource(resource_id, current_user.current_tenant_id, self.resource_model)
        if not current_user.is_editor:
            raise Forbidden()

        current_key_count = (
            db.session.query(ApiToken)
            .filter(ApiToken.type == self.resource_type, getattr(ApiToken, self.resource_id_field) == resource_id)
            .count()
        )

        # cdg:如果当前API密钥数量大于等于最大API密钥数量，则抛出异常。
        if current_key_count >= self.max_keys:
            flask_restful.abort(
                400,
                message=f"Cannot create more than {self.max_keys} API keys for this resource type.",
                code="max_keys_exceeded",
            )

        # cdg:生成API密钥。
        key = ApiToken.generate_api_key(self.token_prefix, 24)
        api_token = ApiToken()
        setattr(api_token, self.resource_id_field, resource_id)
        api_token.tenant_id = current_user.current_tenant_id
        api_token.token = key
        api_token.type = self.resource_type
        db.session.add(api_token)
        db.session.commit()
        return api_token, 201

# cdg:API密钥资源
class BaseApiKeyResource(Resource):
    # cdg:API密钥资源所支持的装饰器
    method_decorators = [account_initialization_required, login_required, setup_required]

    resource_type: str | None = None
    resource_model: Any = None
    resource_id_field: str | None = None

    # cdg:删除API密钥
    def delete(self, resource_id, api_key_id):
        assert self.resource_id_field is not None, "resource_id_field must be set"
        resource_id = str(resource_id)
        api_key_id = str(api_key_id)
        _get_resource(resource_id, current_user.current_tenant_id, self.resource_model)

        # The role of the current user in the ta table must be admin or owner
        if not current_user.is_admin_or_owner:
            raise Forbidden()

        key = (
            db.session.query(ApiToken)
            .filter(
                getattr(ApiToken, self.resource_id_field) == resource_id,
                ApiToken.type == self.resource_type,
                ApiToken.id == api_key_id,
            )
            .first()
        )

        if key is None:
            flask_restful.abort(404, message="API key not found")

        db.session.query(ApiToken).filter(ApiToken.id == api_key_id).delete()
        db.session.commit()

        return {"result": "success"}, 204

# cdg:应用API密钥列表资源
class AppApiKeyListResource(BaseApiKeyListResource):
    def after_request(self, resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp

    resource_type = "app"
    resource_model = App
    resource_id_field = "app_id"
    token_prefix = "app-"

# cdg:应用API密钥资源
class AppApiKeyResource(BaseApiKeyResource):
    def after_request(self, resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp

    resource_type = "app"
    resource_model = App
    resource_id_field = "app_id"

# cdg:知识库API密钥列表资源
class DatasetApiKeyListResource(BaseApiKeyListResource):
    def after_request(self, resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp

    resource_type = "dataset"
    resource_model = Dataset
    resource_id_field = "dataset_id"
    token_prefix = "ds-"

# cdg:知识库API密钥资源
class DatasetApiKeyResource(BaseApiKeyResource):
    def after_request(self, resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        return resp

    resource_type = "dataset"
    resource_model = Dataset
    resource_id_field = "dataset_id"

# cdg:根据应用ID获取API密钥列表资源
api.add_resource(AppApiKeyListResource, "/apps/<uuid:resource_id>/api-keys")
# cdg:根据应用ID获取API密钥资源
api.add_resource(AppApiKeyResource, "/apps/<uuid:resource_id>/api-keys/<uuid:api_key_id>")
# cdg:根据知识库ID获取API密钥列表资源
api.add_resource(DatasetApiKeyListResource, "/datasets/<uuid:resource_id>/api-keys")
# cdg:根据知识库ID获取API密钥资源
api.add_resource(DatasetApiKeyResource, "/datasets/<uuid:resource_id>/api-keys/<uuid:api_key_id>")
