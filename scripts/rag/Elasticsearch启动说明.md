# Elasticsearch 启动说明

## 📋 问题说明

如果遇到 "Elasticsearch连接失败" 或 "连接被意外关闭" 的错误，说明 Elasticsearch 服务未启动。

## ✅ 解决方案

### 方案1：使用 Docker 启动（推荐，最简单）

#### 步骤1：检查 Docker 是否安装

在 PowerShell 中运行：
```powershell
docker --version
```

如果没有安装 Docker，请访问：https://www.docker.com/products/docker-desktop/

#### 步骤2：启动 Elasticsearch 容器

```powershell
# 拉取 Elasticsearch 镜像（首次运行需要）
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# 启动 Elasticsearch 容器
docker run -d `
  --name elasticsearch `
  -p 9200:9200 `
  -p 9300:9300 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" `
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

**注意**：
- `-Xms512m -Xmx512m` 设置 Java 堆内存为 512MB（适合低配置机器）
- 如果机器内存充足，可以改为 `-Xms1g -Xmx1g` 或更大

#### 步骤3：验证 Elasticsearch 是否启动成功

在 PowerShell 中运行：
```powershell
# 方法1：使用 Invoke-WebRequest（PowerShell 原生命令）
Invoke-WebRequest -Uri http://localhost:9200 -UseBasicParsing

# 方法2：在浏览器中访问
# 打开浏览器，访问：http://localhost:9200
```

如果看到类似以下内容，说明启动成功：
```json
{
  "name" : "...",
  "cluster_name" : "docker-cluster",
  "version" : { ... }
}
```

#### 步骤4：查看容器状态

```powershell
# 查看运行中的容器
docker ps

# 查看 Elasticsearch 日志
docker logs elasticsearch

# 如果容器未运行，查看所有容器（包括已停止的）
docker ps -a
```

#### 常用 Docker 命令

```powershell
# 停止 Elasticsearch 容器
docker stop elasticsearch

# 启动已存在的容器
docker start elasticsearch

# 删除容器（注意：会删除所有数据）
docker rm -f elasticsearch

# 重启容器
docker restart elasticsearch
```

---

### 方案2：本地安装 Elasticsearch

#### 步骤1：下载 Elasticsearch

访问：https://www.elastic.co/downloads/elasticsearch

下载 Windows 版本（ZIP 文件）

#### 步骤2：解压并配置

1. 解压到某个目录，例如：`D:\elasticsearch`
2. 编辑 `config/elasticsearch.yml`，添加：
   ```yaml
   discovery.type: single-node
   xpack.security.enabled: false
   ```

#### 步骤3：启动 Elasticsearch

在 PowerShell 中，进入 Elasticsearch 目录：
```powershell
cd D:\elasticsearch
.\bin\elasticsearch.bat
```

---

### 方案3：使用本地文件保存（无需 Elasticsearch）

如果暂时无法启动 Elasticsearch，脚本会自动将向量数据保存到本地文件：

**保存位置**：
- `data/vector_db/policy/policy_vector_db.json` - 政策类向量库（JSON格式）
- `data/vector_db/policy/policy_embeddings.npy` - 政策类向量（NumPy格式）
- `data/vector_db/system/system_vector_db.json` - 系统类向量库（JSON格式）
- `data/vector_db/system/system_embeddings.npy` - 系统类向量（NumPy格式）

**优点**：
- ✅ 无需启动 Elasticsearch
- ✅ 数据完整保存（包含文本、向量、元数据）
- ✅ 可以后续导入到 Elasticsearch

**缺点**：
- ❌ 无法直接进行向量搜索
- ❌ 需要手动加载数据到 Elasticsearch 才能使用

---

## 🔍 故障排查

### 问题1：端口 9200 被占用

**错误信息**：`bind: address already in use`

**解决方法**：
```powershell
# 查找占用 9200 端口的进程
netstat -ano | findstr :9200

# 停止占用端口的进程（替换 PID 为实际进程ID）
taskkill /PID <PID> /F

# 或者修改配置文件中的端口
# 编辑 config/elasticsearch.py，将 9200 改为其他端口，如 9201
```

### 问题2：内存不足

**错误信息**：`max virtual memory areas vm.max_map_count [65530] is too low`

**解决方法**（Windows）：
- 减小 Elasticsearch 的 Java 堆内存设置
- 使用 Docker 时，添加 `-e "ES_JAVA_OPTS=-Xms256m -Xmx256m"`

### 问题3：Docker 容器启动失败

**查看日志**：
```powershell
docker logs elasticsearch
```

**常见原因**：
- 内存不足：减小 Java 堆内存
- 端口被占用：修改端口映射
- 权限问题：以管理员身份运行

---

## 📝 测试连接

### 在 PowerShell 中测试

```powershell
# 方法1：使用 Invoke-WebRequest
$response = Invoke-WebRequest -Uri http://localhost:9200 -UseBasicParsing
$response.Content

# 方法2：使用 curl.exe（如果安装了 Git for Windows）
curl.exe http://localhost:9200

# 方法3：在浏览器中打开
# http://localhost:9200
```

### 在 Python 中测试

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(['localhost:9200'])
if es.ping():
    print("✓ Elasticsearch 连接成功")
    print(es.info())
else:
    print("✗ Elasticsearch 连接失败")
```

---

## 💡 推荐配置

### 低配置机器（内存 < 8GB）

```powershell
docker run -d `
  --name elasticsearch `
  -p 9200:9200 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  -e "ES_JAVA_OPTS=-Xms256m -Xmx256m" `
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

### 中等配置机器（内存 8-16GB）

```powershell
docker run -d `
  --name elasticsearch `
  -p 9200:9200 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" `
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

### 高配置机器（内存 > 16GB）

```powershell
docker run -d `
  --name elasticsearch `
  -p 9200:9200 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" `
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

---

## 🎯 下一步

启动 Elasticsearch 后，重新运行向量库生成脚本：

```powershell
python scripts/rag/build_vector_db.py
```

如果 Elasticsearch 仍然无法连接，脚本会自动保存到本地文件，您可以稍后再启动 Elasticsearch 并导入数据。

