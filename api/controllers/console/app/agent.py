from flask_restful import Resource, reqparse  # type: ignore

from controllers.console import api
from controllers.console.app.wraps import get_app_model
from controllers.console.wraps import account_initialization_required, setup_required
from libs.helper import uuid_value
from libs.login import login_required
from models.model import AppMode
from services.agent_service import AgentService

# cdg: 获取agent日志
class AgentLogApi(Resource):

    # cdg: 获取agent日志
    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model(mode=[AppMode.AGENT_CHAT]) # cdg:获取agent日志接口需要指定应用模式为AGENT_CHAT
    def get(self, app_model):
        """Get agent logs"""
        parser = reqparse.RequestParser()
        # cdg: 解析请求参数，message_id和conversation_id是必传参数
        parser.add_argument("message_id", type=uuid_value, required=True, location="args")
        parser.add_argument("conversation_id", type=uuid_value, required=True, location="args")

        args = parser.parse_args()

        return AgentService.get_agent_logs(app_model, args["conversation_id"], args["message_id"])

# cdg: 添加agent日志API资源
api.add_resource(AgentLogApi, "/apps/<uuid:app_id>/agent/logs")
# cdg:上述接口调用示例：http://127.0.0.1:5000/api/v1/apps/123e4567-e89b-12d3-a456-426614174000/agent/logs?message_id=123e4567-e89b-12d3-a456-426614174000&conversation_id=123e4567-e89b-12d3-a456-426614174000
# 其中123e4567-e89b-12d3-a456-426614174000是app_id，123e4567-e89b-12d3-a456-426614174000是message_id，123e4567-e89b-12d3-a456-426614174000是conversation_id
# 利用curl命令调用示例：curl -X GET "http://127.0.0.1:5000/api/v1/apps/123e4567-e89b-12d3-a456-426614174000/agent/logs?message_id=123e4567-e89b-12d3-a456-426614174000&conversation_id=123e4567-e89b-12d3-a456-426614174000"
# 如果是使用python请求，可以使用requests库，示例代码如下：
# import requests
# response = requests.get("http://127.0.0.1:5000/api/v1/apps/123e4567-e89b-12d3-a456-426614174000/agent/logs?message_id=123e4567-e89b-12d3-a456-426614174000&conversation_id=123e4567-e89b-12d3-a456-426614174000")
# print(response.json())
# 参数传输的另一种方式：curl -X GET "http://127.0.0.1:5000/api/v1/apps/123e4567-e89b-12d3-a456-426614174000/agent/logs?message_id=123e4567-e89b-12d3-a456-426614174000&conversation_id=123e4567-e89b-12d3-a456-426614174000" -H "Content-Type: application/json" -d '{"message_id": "123e4567-e89b-12d3-a456-426614174000", "conversation_id": "123e4567-e89b-12d3-a456-426614174000"}'
# 如果是使用python请求，可以使用requests库，示例代码如下：
# import requests
# response = requests.get("http://127.0.0.1:5000/api/v1/apps/123e4567-e89b-12d3-a456-426614174000/agent/logs?message_id=123e4567-e89b-12d3-a456-426614174000&conversation_id=123e4567-e89b-12d3-a456-426614174000", headers={"Content-Type": "application/json"}, data='{"message_id": "123e4567-e89b-12d3-a456-426614174000", "conversation_id": "123e4567-e89b-12d3-a456-426614174000"}')
# print(response.json())


