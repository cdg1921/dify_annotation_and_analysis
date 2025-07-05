from dify_app import DifyApp 

# cdg: 初始化命令，这个函数核心作用是批量注册一组管理命令到DifyApp的CLI。
# 每个命令都对应一个具体的管理操作，比如重置密码、升级数据库、修复站点等。
# 这样做的好处是，后续只需调用init_app(app)，就能一次性把所有常用的管理命令集成到CLI工具中，方便维护和运维人员使用。
# 用户也可以在命令行中直接调用这些命令，比如：
# flask reset-password --email=admin@example.com --new-password=newpassword --password-confirm=newpassword
# flask upgrade-db
# flask fix-app-site-missing
# flask reset-email --email=admin@example.com
# flask reset-encrypt-key-pair
# flask vdb-migrate
def init_app(app: DifyApp):
    from commands import (
        add_qdrant_doc_id_index, #cdg: 添加Qdrant文档ID索引
        convert_to_agent_apps, #cdg: 转换为代理应用
        create_tenant, #cdg: 创建租户
        fix_app_site_missing, #cdg: 修复应用站点缺失
        reset_email, #cdg: 重置邮箱
        reset_encrypt_key_pair, #cdg: 重置加密密钥对
        reset_password, #cdg: 重置密码
        upgrade_db, #cdg: 升级数据库
        vdb_migrate, #cdg: 迁移数据库
    )
    # cdg: 注册命令
    cmds_to_register = [
        reset_password, #cdg: 重置密码
        reset_email, #cdg: 重置邮箱
        reset_encrypt_key_pair, #cdg: 重置加密密钥对
        vdb_migrate, #cdg: 迁移数据库
        convert_to_agent_apps, #cdg: 转换为代理应用
        add_qdrant_doc_id_index, #cdg: 添加Qdrant文档ID索引
        create_tenant, #cdg: 创建租户
        upgrade_db, #cdg: 升级数据库
        fix_app_site_missing, #cdg: 修复应用站点缺失    
    ]
    for cmd in cmds_to_register:
        app.cli.add_command(cmd)
