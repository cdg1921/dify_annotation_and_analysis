import uuid
from typing import Optional

from flask_login import current_user  # type: ignore
from sqlalchemy import func
from werkzeug.exceptions import NotFound

from extensions.ext_database import db
from models.dataset import Dataset
from models.model import App, Tag, TagBinding

# cdg: 标签管理服务，实现标签的获取、保存、更新、删除和绑定。
class TagService:
    # cdg: 获取标签。
    @staticmethod
    def get_tags(tag_type: str, current_tenant_id: str, keyword: Optional[str] = None) -> list:
        query = (
            db.session.query(Tag.id, Tag.type, Tag.name, func.count(TagBinding.id).label("binding_count"))
            .outerjoin(TagBinding, Tag.id == TagBinding.tag_id)
            .filter(Tag.type == tag_type, Tag.tenant_id == current_tenant_id)
        )
        if keyword:
            query = query.filter(db.and_(Tag.name.ilike(f"%{keyword}%")))
        query = query.group_by(Tag.id)
        results: list = query.order_by(Tag.created_at.desc()).all()
        return results

    # cdg: 根据标签ID获取目标ID，具体实现为：
    # 1. 根据标签ID获取标签。
    # 2. 根据标签ID获取目标ID。
    # 3. 返回目标ID列表。
    @staticmethod
    def get_target_ids_by_tag_ids(tag_type: str, current_tenant_id: str, tag_ids: list) -> list:
        tags = (
            db.session.query(Tag)
            .filter(Tag.id.in_(tag_ids), Tag.tenant_id == current_tenant_id, Tag.type == tag_type)
            .all()
        )
        if not tags:
            return []
        tag_ids = [tag.id for tag in tags]
        tag_bindings = (
            db.session.query(TagBinding.target_id)
            .filter(TagBinding.tag_id.in_(tag_ids), TagBinding.tenant_id == current_tenant_id)
            .all()
        )
        if not tag_bindings:
            return []
        results = [tag_binding.target_id for tag_binding in tag_bindings]
        return results

    # cdg: 根据目标ID获取标签。
    @staticmethod
    def get_tags_by_target_id(tag_type: str, current_tenant_id: str, target_id: str) -> list:
        tags = (
            db.session.query(Tag)
            .join(TagBinding, Tag.id == TagBinding.tag_id)
            .filter(
                TagBinding.target_id == target_id,
                TagBinding.tenant_id == current_tenant_id,
                Tag.tenant_id == current_tenant_id,
                Tag.type == tag_type,
            )
            .all()
        )

        return tags or []

    # cdg: 保存标签。
    @staticmethod
    def save_tags(args: dict) -> Tag:
        tag = Tag(
            id=str(uuid.uuid4()),
            name=args["name"],
            type=args["type"],
            created_by=current_user.id,
            tenant_id=current_user.current_tenant_id,
        )
        db.session.add(tag)
        db.session.commit()
        return tag

    # cdg: 更新标签。
    @staticmethod
    def update_tags(args: dict, tag_id: str) -> Tag:
        tag = db.session.query(Tag).filter(Tag.id == tag_id).first()
        if not tag:
            raise NotFound("Tag not found")
        tag.name = args["name"]
        db.session.commit()
        return tag

    # cdg: 获取标签绑定数量。
    @staticmethod
    def get_tag_binding_count(tag_id: str) -> int:
        count = db.session.query(TagBinding).filter(TagBinding.tag_id == tag_id).count()
        return count

    # cdg: 删除标签。
    @staticmethod
    def delete_tag(tag_id: str):
        tag = db.session.query(Tag).filter(Tag.id == tag_id).first()
        if not tag:
            raise NotFound("Tag not found")
        db.session.delete(tag)
        # delete tag binding
        tag_bindings = db.session.query(TagBinding).filter(TagBinding.tag_id == tag_id).all()
        if tag_bindings:
            for tag_binding in tag_bindings:
                db.session.delete(tag_binding)
        db.session.commit()

    # cdg: 保存标签绑定。
    @staticmethod
    def save_tag_binding(args):
        # check if target exists
        TagService.check_target_exists(args["type"], args["target_id"])
        # save tag binding
        for tag_id in args["tag_ids"]:
            tag_binding = (
                db.session.query(TagBinding)
                .filter(TagBinding.tag_id == tag_id, TagBinding.target_id == args["target_id"])
                .first()
            )
            if tag_binding:
                continue
            new_tag_binding = TagBinding(
                tag_id=tag_id,
                target_id=args["target_id"],
                tenant_id=current_user.current_tenant_id,
                created_by=current_user.id,
            )
            db.session.add(new_tag_binding)
        db.session.commit()

    # cdg: 删除标签绑定。
    @staticmethod
    def delete_tag_binding(args):
        # check if target exists
        TagService.check_target_exists(args["type"], args["target_id"])
        # delete tag binding
        tag_bindings = (
            db.session.query(TagBinding)
            .filter(TagBinding.target_id == args["target_id"], TagBinding.tag_id == (args["tag_id"]))
            .first()
        )
        if tag_bindings:
            db.session.delete(tag_bindings)
            db.session.commit()

    # cdg: 检查目标是否存在。
    @staticmethod
    def check_target_exists(type: str, target_id: str):
        if type == "knowledge":
            dataset = (
                db.session.query(Dataset)
                .filter(Dataset.tenant_id == current_user.current_tenant_id, Dataset.id == target_id)
                .first()
            )
            if not dataset:
                raise NotFound("Dataset not found")
        elif type == "app":
            app = (
                db.session.query(App)
                .filter(App.tenant_id == current_user.current_tenant_id, App.id == target_id)
                .first()
            )
            if not app:
                raise NotFound("App not found")
        else:
            raise NotFound("Invalid binding type")
