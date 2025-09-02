import os
# cdg: 获取当前文件的绝对路径
# Getting the absolute path of the current file's directory
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# cdg: 获取项目根目录的绝对路径
# Getting the absolute path of the project's root directory
PROJECT_DIR = os.path.abspath(os.path.join(ABS_PATH, os.pardir, os.pardir))

# cdg: 加载.env文件
# Loading the .env file if it exists
def _load_env() -> None:
    dotenv_path = os.path.join(PROJECT_DIR, "tests", "integration_tests", ".env")
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        # cdg: 加载.env文件
        load_dotenv(dotenv_path)


_load_env()
