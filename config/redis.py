"""
Redis连接配置模块
用于管理Redis连接配置信息，用于L2业务状态记忆的快速查询和Session管理

注意：
1. 如果Redis未部署，请将REDIS_ENABLED设置为False（默认已关闭）
2. Redis关闭时，系统会降级运行：
   - Session管理：仍可正常工作，但不会持久化到Redis
   - L2记忆：仍可正常工作，但只保存到ES（审计），不保存到Redis（快速查询）
3. 启用Redis时，请确保Redis服务已启动并可连接
"""

# ============================================================================
# Redis功能开关
# ============================================================================

# Redis功能是否启用（默认False，关闭状态）
# 如果Redis未部署，请保持为False，系统会降级运行（不影响主流程）
REDIS_ENABLED = False

# ============================================================================
# Redis连接配置（使用默认配置）
# ============================================================================

# Redis服务器地址配置
REDIS_HOST = "localhost"          # Redis服务器地址，如果Redis不在本地，请修改为实际IP（如：192.168.1.100）
REDIS_PORT = 6379                 # Redis端口，默认6379
REDIS_DB = 0                      # Redis数据库编号，默认0
REDIS_PASSWORD = None             # Redis密码，如果未设置密码则为None

# Redis连接配置字典
REDIS_CONFIG = {
    'host': REDIS_HOST,
    'port': REDIS_PORT,
    'db': REDIS_DB,
    'decode_responses': True,      # 自动解码响应为字符串
    'socket_timeout': 5,           # Socket超时时间（秒）
    'socket_connect_timeout': 5,   # Socket连接超时时间（秒）
    'retry_on_timeout': True,      # 超时时重试
    'health_check_interval': 30,   # 健康检查间隔（秒）
}

# 如果配置了密码，添加到配置中
if REDIS_PASSWORD:
    REDIS_CONFIG['password'] = REDIS_PASSWORD

# ============================================================================
# Redis Key前缀配置
# ============================================================================

# L2业务状态记忆的Key前缀
# 格式：l2_memory:{session_id}
L2_MEMORY_KEY_PREFIX = "l2_memory"

# Session信息的Key前缀
# 格式：session:{session_id}
SESSION_KEY_PREFIX = "session"

# ============================================================================
# Redis过期时间配置（秒）
# ============================================================================

# L2记忆的过期时间（默认24小时，86400秒）
# 注意：L2记忆的生命周期同session，session过期时L2也会过期
L2_MEMORY_TTL = 86400

# Session的过期时间（默认24小时，86400秒）
SESSION_TTL = 86400

