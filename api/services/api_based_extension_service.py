from core.extension.api_based_extension_requestor import APIBasedExtensionRequestor
from core.helper.encrypter import decrypt_token, encrypt_token
from extensions.ext_database import db
from models.api_based_extension import APIBasedExtension, APIBasedExtensionPoint

# cdg: APIBasedExtension服务，包括获取所有APIBasedExtension、保存APIBasedExtension、删除APIBasedExtension、获取APIBasedExtension、验证APIBasedExtension。
class APIBasedExtensionService:
    # cdg: 获取所有APIBasedExtension，具体实现为：从数据库中获取所有APIBasedExtension；如果APIBasedExtension不存在，则抛出异常；返回APIBasedExtension。
    @staticmethod
    def get_all_by_tenant_id(tenant_id: str) -> list[APIBasedExtension]:
        extension_list = (
            db.session.query(APIBasedExtension)
            .filter_by(tenant_id=tenant_id)
            .order_by(APIBasedExtension.created_at.desc())
            .all()
        )

        for extension in extension_list:
            extension.api_key = decrypt_token(extension.tenant_id, extension.api_key)

        return extension_list

    # cdg: 保存APIBasedExtension，具体实现为：验证APIBasedExtension；加密APIBasedExtension的APIKey；提交数据库。
    @classmethod
    def save(cls, extension_data: APIBasedExtension) -> APIBasedExtension:
        cls._validation(extension_data)

        extension_data.api_key = encrypt_token(extension_data.tenant_id, extension_data.api_key)

        db.session.add(extension_data)
        db.session.commit()
        return extension_data

    # cdg: 删除APIBasedExtension，具体实现为：删除APIBasedExtension；提交数据库。
    @staticmethod
    def delete(extension_data: APIBasedExtension) -> None:
        db.session.delete(extension_data)
        db.session.commit()

    # cdg: 获取APIBasedExtension，具体实现为：从数据库中获取APIBasedExtension；如果APIBasedExtension不存在，则抛出异常；返回APIBasedExtension。
    @staticmethod
    def get_with_tenant_id(tenant_id: str, api_based_extension_id: str) -> APIBasedExtension:
        extension = (
            db.session.query(APIBasedExtension)
            .filter_by(tenant_id=tenant_id)
            .filter_by(id=api_based_extension_id)
            .first()
        )

        if not extension:
            raise ValueError("API based extension is not found")

        extension.api_key = decrypt_token(extension.tenant_id, extension.api_key)

        return extension

    # cdg: 验证APIBasedExtension，具体实现为：验证APIBasedExtension的名称、API端点、APIKey。
    @classmethod
    def _validation(cls, extension_data: APIBasedExtension) -> None:
        # name
        if not extension_data.name:
            raise ValueError("name must not be empty")

        if not extension_data.id:
            # case one: check new data, name must be unique
            is_name_existed = (
                db.session.query(APIBasedExtension)
                .filter_by(tenant_id=extension_data.tenant_id)
                .filter_by(name=extension_data.name)
                .first()
            )

            if is_name_existed:
                raise ValueError("name must be unique, it is already existed")
        else:
            # case two: check existing data, name must be unique
            is_name_existed = (
                db.session.query(APIBasedExtension)
                .filter_by(tenant_id=extension_data.tenant_id)
                .filter_by(name=extension_data.name)
                .filter(APIBasedExtension.id != extension_data.id)
                .first()
            )

            if is_name_existed:
                raise ValueError("name must be unique, it is already existed")

        # cdg: 验证API端点
        # api_endpoint
        if not extension_data.api_endpoint:
            raise ValueError("api_endpoint must not be empty")

        # cdg: 验证APIKey
        # api_key
        if not extension_data.api_key:
            raise ValueError("api_key must not be empty")

        if len(extension_data.api_key) < 5:
            raise ValueError("api_key must be at least 5 characters")

        # cdg: 验证API端点
        # check endpoint
        cls._ping_connection(extension_data)

    # cdg: 验证API端点，具体实现为：创建APIBasedExtensionRequestor；发送请求；如果请求结果不等于pong，则抛出异常。
    @staticmethod
    def _ping_connection(extension_data: APIBasedExtension) -> None:
        try:
            client = APIBasedExtensionRequestor(extension_data.api_endpoint, extension_data.api_key)
            resp = client.request(point=APIBasedExtensionPoint.PING, params={})
            if resp.get("result") != "pong":
                raise ValueError(resp)
        except Exception as e:
            raise ValueError("connection error: {}".format(e))
