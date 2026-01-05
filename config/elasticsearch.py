"""
Elasticsearch连接配置模块
用于管理Elasticsearch连接配置信息
"""

# ============================================================================
# Elasticsearch连接配置
# ============================================================================

# 服务器地址配置
ES_HOST = "localhost"          # ES服务器地址，如果ES不在本地，请修改为实际IP（如：192.168.1.100）
ES_PORT = 9200                 # ES端口，默认9200
ES_PROTOCOL = "https"          # 协议类型：http 或 https

# 认证配置（如果ES启用了安全认证，请填写用户名和密码；如果不需要认证，设为None）
ES_USERNAME = "elastic"         # ES用户名
ES_PASSWORD = "elastic"         # ES密码（请根据实际情况修改）

# 构建连接URL
es_url = f"{ES_PROTOCOL}://{ES_HOST}:{ES_PORT}"

# Elasticsearch客户端配置
# 注意：Elasticsearch 9.x版本使用 basic_auth 而不是 http_auth
ES_CONFIG = {
    'hosts': [es_url],                    # ES服务器地址列表
    'request_timeout': 30,                # 请求超时时间（秒）
    'max_retries': 2,                     # 最大重试次数
    'retry_on_timeout': True,             # 超时是否重试
    'verify_certs': False,                # 是否验证SSL证书（HTTPS自签名证书时设为False）
    'ssl_show_warn': False,               # 是否显示SSL警告
}

# 添加认证信息（如果配置了用户名和密码）
if ES_USERNAME and ES_PASSWORD:
    ES_CONFIG['basic_auth'] = (ES_USERNAME, ES_PASSWORD)

# ============================================================================
# 索引配置
# ============================================================================

# 索引配置字典，包含政策类和系统功能类的索引配置
INDEX_CONFIG = {
    'policy': {
        'index_name': 'bank_credit_policy',    # 政策类索引名称
        'vector_dimension': 1024,             # Qwen3-Embedding向量维度
    },
    'system': {
        'index_name': 'bank_credit_system',   # 系统功能类索引名称
        'vector_dimension': 1024,             # Qwen3-Embedding向量维度
    },
    'l1_memory': {
        'index_name': 'bank_credit_l1_memory',  # L1会话级记忆索引名称
    },
    'l2_memory': {
        'index_name': 'bank_credit_l2_memory',  # L2业务状态记忆索引名称
    }
}

# ============================================================================
# 批量操作配置
# ============================================================================

# 批量插入时每批的文档数量
BATCH_SIZE = 100

