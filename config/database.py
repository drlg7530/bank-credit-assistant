"""
数据库连接配置模块
用于管理MySQL数据库连接配置信息
"""

# MySQL数据库连接配置
DB_CONFIG = {
    'host': 'localhost',        # 数据库主机地址
    'port': 3306,               # 数据库端口号
    'user': 'root',             # 数据库用户名
    'password': 'root',         # 数据库密码
    'database': 'bank_credit_agent',  # 数据库名称
    'charset': 'utf8mb4',       # 字符集，支持中文和特殊字符
    'autocommit': False,        # 是否自动提交事务
    'connect_timeout': 10,      # 连接超时时间（秒）
}

# 数据库连接池配置（可选，用于高并发场景）
POOL_CONFIG = {
    'pool_name': 'bank_credit_pool',  # 连接池名称
    'pool_size': 5,                   # 连接池大小
    'pool_reset_session': True,       # 重置会话
}

