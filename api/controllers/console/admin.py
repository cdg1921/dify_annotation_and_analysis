from functools import wraps

from flask import request
from flask_restful import Resource, reqparse  # type: ignore
from werkzeug.exceptions import NotFound, Unauthorized

from configs import dify_config
from constants.languages import supported_language
from controllers.console import api
from controllers.console.wraps import only_edition_cloud
from extensions.ext_database import db
from models.model import App, InstalledApp, RecommendedApp

# cdg:管理员权限必备装饰器，用于装饰视图函数，使要求具有管理员权限才能访问。
def admin_required(view):
    # cdg:@wraps(view) 用于装饰视图函数，使其具有装饰器的功能。参数view是视图函数，即admin_required装饰器装饰的函数。
    @wraps(view)
    def decorated(*args, **kwargs):
        # cdg:如果管理员API密钥不存在，则抛出异常。
        if not dify_config.ADMIN_API_KEY:
            raise Unauthorized("API key is invalid.")
        # cdg:获取请求头中的Authorization字段。
        auth_header = request.headers.get("Authorization")
        if auth_header is None:
            raise Unauthorized("Authorization header is missing.")
        # cdg:如果Authorization字段中不包含空格，则抛出异常。是Bearer <api-key>格式。
        if " " not in auth_header:
            raise Unauthorized("Invalid Authorization header format. Expected 'Bearer <api-key>' format.")
        # cdg:如果Authorization字段中包含空格，则分割出授权方案和授权令牌。
        auth_scheme, auth_token = auth_header.split(None, 1)
        auth_scheme = auth_scheme.lower()
        # cdg:如果授权方案不是Bearer，则抛出异常。
        if auth_scheme != "bearer":
            raise Unauthorized("Invalid Authorization header format. Expected 'Bearer <api-key>' format.")
        # cdg:如果授权令牌不是管理员API密钥，则抛出异常。
        if auth_token != dify_config.ADMIN_API_KEY:
            raise Unauthorized("API key is invalid.")

        return view(*args, **kwargs)

    # cdg:返回装饰后的视图函数。
    return decorated

# cdg:插入探索应用列表
class InsertExploreAppListApi(Resource):
    @only_edition_cloud  # cdg:只允许云版使用。
    @admin_required  # cdg:要求具有管理员权限才能访问。
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("app_id", type=str, required=True, nullable=False, location="json")
        parser.add_argument("desc", type=str, location="json")
        parser.add_argument("copyright", type=str, location="json")
        parser.add_argument("privacy_policy", type=str, location="json")
        parser.add_argument("custom_disclaimer", type=str, location="json")
        parser.add_argument("language", type=supported_language, required=True, nullable=False, location="json")
        parser.add_argument("category", type=str, required=True, nullable=False, location="json")
        parser.add_argument("position", type=int, required=True, nullable=False, location="json")
        args = parser.parse_args()

        # cdg:根据app_id查询应用。
        app = App.query.filter(App.id == args["app_id"]).first()
        if not app:
            raise NotFound(f'App \'{args["app_id"]}\' is not found')

        # cdg:根据应用的site查询应用的描述、版权、隐私政策、自定义免责声明。site是应用的站点。
        site = app.site
        if not site:
            desc = args["desc"] or ""
            copy_right = args["copyright"] or ""
            privacy_policy = args["privacy_policy"] or ""
            custom_disclaimer = args["custom_disclaimer"] or ""
        else:
            desc = site.description or args["desc"] or ""
            copy_right = site.copyright or args["copyright"] or ""
            privacy_policy = site.privacy_policy or args["privacy_policy"] or ""
            custom_disclaimer = site.custom_disclaimer or args["custom_disclaimer"] or ""

        # cdg:根据app_id查询推荐应用。
        recommended_app = RecommendedApp.query.filter(RecommendedApp.app_id == args["app_id"]).first()

        # cdg:如果推荐应用不存在，则创建推荐应用，并设置应用的描述、版权、隐私政策、自定义免责声明。
        if not recommended_app:
            recommended_app = RecommendedApp(
                app_id=app.id,
                description=desc,
                copyright=copy_right,
                privacy_policy=privacy_policy,
                custom_disclaimer=custom_disclaimer,
                language=args["language"],
                category=args["category"],
                position=args["position"],
            )

            db.session.add(recommended_app)

            app.is_public = True
            db.session.commit()

            return {"result": "success"}, 201
        else:
            # cdg:如果推荐应用存在，则更新推荐应用的描述、版权、隐私政策、自定义免责声明。
            recommended_app.description = desc
            recommended_app.copyright = copy_right
            recommended_app.privacy_policy = privacy_policy
            recommended_app.custom_disclaimer = custom_disclaimer
            recommended_app.language = args["language"]
            recommended_app.category = args["category"]
            recommended_app.position = args["position"]

            app.is_public = True

            db.session.commit()

            return {"result": "success"}, 200


# cdg:插入探索应用
class InsertExploreAppApi(Resource):
    @only_edition_cloud  # cdg:只允许云版使用。
    @admin_required  # cdg:要求具有管理员权限才能访问。
    def delete(self, app_id):
        # cdg:根据app_id查询推荐应用。
        recommended_app = RecommendedApp.query.filter(RecommendedApp.app_id == str(app_id)).first()
        if not recommended_app:
            return {"result": "success"}, 204
        # cdg:根据推荐应用的app_id查询应用。
        app = App.query.filter(App.id == recommended_app.app_id).first()
        if app:
            app.is_public = False
        # cdg:根据推荐应用的app_id查询已安装应用。
        installed_apps = InstalledApp.query.filter(
            InstalledApp.app_id == recommended_app.app_id, InstalledApp.tenant_id != InstalledApp.app_owner_tenant_id
        ).all()
        # cdg:删除已安装应用。
        for installed_app in installed_apps:
            db.session.delete(installed_app)

        db.session.delete(recommended_app)
        db.session.commit()

        return {"result": "success"}, 204

# cdg:插入探索应用列表接口路由。
api.add_resource(InsertExploreAppListApi, "/admin/insert-explore-apps")
# cdg:插入探索应用接口路由。
api.add_resource(InsertExploreAppApi, "/admin/insert-explore-apps/<uuid:app_id>")
