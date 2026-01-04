"""
RAG向量库生成脚本
功能：
1. 读取QA对JSON文件（data/qa_pairs/）
2. 将QA对转换为文档块格式
3. 生成Embedding向量
4. 保存到Elasticsearch（单个混合索引，同时包含向量和文本字段，支持ES原生Hybrid Search）

Metadata设计（符合需求文档）：
- domain: 域类型（政策 | 系统）
- doc_type: 文档类型（监管政策 | 内部制度 | 系统说明 | 操作手册）
- source: 来源机构（国家金融监督管理总局等）
- region: 地区范围（全国 | 新疆 等）
- publish_date: 发布时间（YYYY-MM-DD格式）
- status: 状态（生效 | 已失效）
- doc_id: 文档编号
- role: 权限角色（客户经理 | 团队负责人 | 行长，用于查询权限控制）
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import hashlib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入Elasticsearch配置
from config.elasticsearch import ES_CONFIG, INDEX_CONFIG, BATCH_SIZE

# ============================================================================
# 配置区域
# ============================================================================

# Embedding模型配置
# 使用ModelScope的Qwen3-Embedding模型
# 可选模型：
# - Qwen/Qwen3-Embedding-0.6B (0.6B参数，速度快，适合开发测试)
# - Qwen/Qwen3-Embedding-8B (8B参数，效果好，适合生产环境)
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')
# ModelScope模型路径格式：Qwen/Qwen3-Embedding-0.6B 或 Qwen/Qwen3-Embedding-8B

# Embedding批处理配置（针对低配置机器优化）
# 如果机器配置较低，可以减小这个值（建议4-8）
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '4'))  # 默认4，低配置机器更友好

# QA对文件路径
QA_PAIRS_DIR = project_root / "data" / "qa_pairs"

# ============================================================================
# 依赖检查
# ============================================================================

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠ 警告: numpy未安装，将无法生成向量库")
    print("   请安装: pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple")

try:
    from modelscope import snapshot_download
    from transformers import AutoTokenizer, AutoModel
    import torch
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("⚠ 警告: modelscope或transformers未安装，将无法生成embedding")
    print("   请安装: pip install modelscope transformers torch -i https://pypi.tuna.tsinghua.edu.cn/simple")

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠ 警告: elasticsearch未安装，将无法保存到Elasticsearch")
    print("   请安装: pip install elasticsearch -i https://pypi.tuna.tsinghua.edu.cn/simple")


# ============================================================================
# Embedding生成函数
# ============================================================================

def load_embedding_model(model_path: str):
    """
    加载ModelScope的Qwen3-Embedding模型
    
    参数:
        model_path: 模型路径（ModelScope格式）
    
    返回:
        tuple: (model, tokenizer)
    """
    print(f"  正在从ModelScope加载模型: {model_path}")
    
    # 下载模型（如果未下载）
    try:
        model_dir = snapshot_download(model_path, cache_dir='./models/embedding')
        print(f"  ✓ 模型路径: {model_dir}")
    except Exception as e:
        print(f"  ⚠ 模型下载失败: {e}")
        raise
    
    # 加载tokenizer和model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 检查是否有GPU可用，如果GPU内存不足，使用CPU
        use_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        if use_cpu:
            print(f"  ℹ 强制使用CPU模式（环境变量FORCE_CPU=true）")
            device_map = 'cpu'
        elif torch.cuda.is_available():
            # 检查GPU内存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"  ℹ 检测到GPU，显存: {gpu_memory:.1f}GB")
            if gpu_memory < 4:  # 如果显存小于4GB，建议使用CPU
                print(f"  ⚠ 显存较小，建议设置环境变量 FORCE_CPU=true 使用CPU模式")
            device_map = 'auto'
        else:
            print(f"  ℹ 未检测到GPU，使用CPU模式")
            device_map = 'cpu'
        
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, device_map=device_map)
        model.eval()
        
        # 显示模型实际使用的设备
        actual_device = next(model.parameters()).device
        print(f"  ✓ 模型加载成功，使用设备: {actual_device}")
        return model, tokenizer
    except Exception as e:
        print(f"  ⚠ 模型加载失败: {e}")
        print(f"    提示: 如果GPU内存不足，可以设置环境变量 FORCE_CPU=true 强制使用CPU")
        raise


def generate_embeddings(texts: List[str], model, tokenizer, batch_size: int = 4) -> np.ndarray:
    """
    使用Qwen3-Embedding生成文本的embedding向量
    
    参数:
        texts: 文本列表
        model: Qwen3-Embedding模型
        tokenizer: 对应的tokenizer
        batch_size: 批处理大小（默认4，适合低配置机器）
    
    返回:
        np.ndarray: embedding向量矩阵
    """
    all_embeddings = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总批次数
    print(f"  正在生成embedding向量（共 {len(texts)} 条文本，分 {total_batches} 批处理，每批 {batch_size} 条）...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 分批处理
    for batch_idx, i in enumerate(range(0, len(texts), batch_size), 1):
        batch_texts = texts[i:i + batch_size]
        batch_start_time = time.time()
        
        try:
            # Tokenize（文本编码）
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到模型设备（GPU或CPU）
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成embedding（模型推理）
            with torch.no_grad():  # 禁用梯度计算，节省内存
                outputs = model(**inputs)
                # Qwen3-Embedding可能返回不同的格式
                # 优先使用pooler_output（如果有），否则使用mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output.cpu().numpy()
                elif hasattr(outputs, 'last_hidden_state'):
                    # 使用mean pooling获取句子级别的embedding
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    # 如果都没有，尝试直接使用outputs
                    embeddings = outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
                    if isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.numpy()
            
            all_embeddings.append(embeddings)
            
            # 计算处理时间
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            
            # 每批都显示进度（让用户知道程序在运行）
            processed_count = min(i + batch_size, len(texts))
            progress = (processed_count / len(texts)) * 100
            avg_time_per_batch = elapsed_time / batch_idx
            estimated_remaining = avg_time_per_batch * (total_batches - batch_idx)
            
            print(f"    [{batch_idx}/{total_batches}] 已处理 {processed_count}/{len(texts)} 条 "
                  f"({progress:.1f}%) | 本批耗时: {batch_time:.1f}秒 | "
                  f"预计剩余: {estimated_remaining:.0f}秒")
            
            # 清理GPU缓存（如果使用GPU）
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"    ⚠ 处理第 {batch_idx} 批时出错: {e}")
            print(f"    提示: 如果内存不足，可以减小 EMBEDDING_BATCH_SIZE 环境变量（当前: {batch_size}）")
            raise
    
    # 合并所有embeddings
    final_embeddings = np.vstack(all_embeddings)
    total_time = time.time() - start_time
    avg_time_per_text = total_time / len(texts)
    
    print(f"  ✓ 生成了 {len(final_embeddings)} 个embedding向量 (维度: {final_embeddings.shape[1]})")
    print(f"  ✓ 总耗时: {total_time:.1f}秒，平均每条: {avg_time_per_text:.2f}秒")
    
    return final_embeddings


# ============================================================================
# Elasticsearch向量库保存函数
# ============================================================================

def test_elasticsearch_connection() -> bool:
    """
    测试Elasticsearch连接是否正常
    
    返回:
        bool: 连接成功返回True，失败返回False
    """
    if not ELASTICSEARCH_AVAILABLE:
        print("⚠ 错误: elasticsearch库未安装")
        print("   请安装: pip install elasticsearch -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return False
    
    print("\n正在测试Elasticsearch连接...")
    print(f"  连接地址: {ES_CONFIG.get('hosts', ['未知'])}")
    print(f"  SSL验证: {ES_CONFIG.get('verify_certs', True)}")
    # 显示认证信息（不显示密码）
    if 'basic_auth' in ES_CONFIG:
        username = ES_CONFIG['basic_auth'][0] if isinstance(ES_CONFIG['basic_auth'], tuple) else '已配置'
        print(f"  认证用户: {username}")
    else:
        print(f"  认证方式: 无认证")
    
    try:
        # 创建Elasticsearch客户端
        es_client = Elasticsearch(**ES_CONFIG)
        
        # 尝试获取集群信息（比ping更可靠）
        try:
            info = es_client.info()
            print(f"  ✓ Elasticsearch连接成功！")
            print(f"    集群名称: {info.get('cluster_name', '未知')}")
            print(f"    版本: {info.get('version', {}).get('number', '未知')}")
            print(f"    节点名称: {info.get('name', '未知')}")
            return True
        except Exception as info_error:
            # 如果info()失败，尝试ping()
            print(f"  ⚠ 获取集群信息失败: {info_error}")
            try:
                if es_client.ping():
                    print(f"  ✓ ping成功，但无法获取详细信息")
                    return True
                else:
                    print(f"  ⚠ ping也失败")
            except Exception as ping_error:
                print(f"  ⚠ ping失败: {ping_error}")
            
            return False
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ⚠ Elasticsearch连接失败: {error_msg}")
        
        # 根据错误类型给出具体建议
        if 'SSL' in error_msg or 'certificate' in error_msg.lower():
            print(f"\n  SSL证书错误，尝试以下解决方案:")
            print(f"    1. 确认ES使用HTTPS（当前配置: {ES_CONFIG.get('hosts')}）")
            print(f"    2. 如果ES使用HTTP，请修改 config/elasticsearch.py:")
            print(f"       'hosts': ['http://localhost:9200']")
            print(f"    3. 如果是自签名证书，确保 verify_certs=False")
        elif 'Connection' in error_msg or 'refused' in error_msg.lower():
            print(f"\n  连接被拒绝，请检查:")
            print(f"    1. Elasticsearch服务是否正在运行")
            print(f"    2. 端口9200是否被占用")
            print(f"    3. 防火墙是否阻止了连接")
        else:
            print(f"\n  请检查:")
            print(f"    1. Elasticsearch服务是否正常运行")
            print(f"    2. 配置中的地址和端口是否正确（当前: {ES_CONFIG.get('hosts', ['未知'])}）")
            print(f"    3. 是否使用了正确的协议（http:// 或 https://）")
            print(f"    4. 如果是HTTPS，证书是否正确（当前verify_certs={ES_CONFIG.get('verify_certs', True)}）")
        
        return False


def create_vector_index(es_client: Elasticsearch, index_name: str, vector_dimension: int):
    """
    创建支持Hybrid Search的混合索引（同时支持向量搜索和文本搜索）
    
    参数:
        es_client: Elasticsearch客户端
        index_name: 索引名称
        vector_dimension: 向量维度
    
    功能:
        - 定义索引映射，包含dense_vector字段用于向量搜索（KNN）
        - 同时包含文本字段用于文本搜索（BM25）
        - 支持ES原生Hybrid Search（BM25 + 向量搜索）
        - 创建索引（如果已存在则删除重建）
    """
    # 检查索引是否已存在，如果存在则删除
    if es_client.indices.exists(index=index_name):
        print(f"  索引 {index_name} 已存在，删除旧索引...")
        es_client.indices.delete(index=index_name)
    
    # 选择分词器（优先使用IK分词器，如果不可用则使用标准分词器）
    analyzer_name = "ik_max_word"
    search_analyzer_name = "ik_max_word"
    
    # 定义索引映射（向量模式，包含向量字段和文本字段）
    index_mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                # 文档标题字段（文本搜索）
                "title": {
                    "type": "text",
                    "analyzer": analyzer_name,
                    "search_analyzer": search_analyzer_name,
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                # 文档源文件路径
                "source": {
                    "type": "keyword"
                },
                # 文档块内容（文本搜索）
                "content": {
                    "type": "text",
                    "analyzer": analyzer_name,
                    "search_analyzer": search_analyzer_name
                },
                # 向量字段（用于向量搜索）
                "embedding": {
                    "type": "dense_vector",      # 密集向量类型
                    "dims": vector_dimension,    # 向量维度
                    "index": True,               # 启用向量索引
                    "similarity": "cosine"       # 使用余弦相似度
                },
                # 块ID
                "chunk_id": {
                    "type": "keyword"
                },
                # 页码
                "page_num": {
                    "type": "integer"
                },
                # Token数量
                "token_count": {
                    "type": "integer"
                },
                # 文档类型
                "file_type": {
                    "type": "keyword"
                },
                # Metadata字段（用于过滤）
                "domain": {
                    "type": "keyword"
                },
                "doc_type": {
                    "type": "keyword"
                },
                "region": {
                    "type": "keyword"
                },
                "publish_date": {
                    "type": "date",
                    "format": "yyyy-MM-dd"
                },
                "status": {
                    "type": "keyword"
                },
                "role": {
                    "type": "keyword"
                },
                "doc_id": {
                    "type": "keyword"
                },
                # QA对相关字段
                "qa_id": {
                    "type": "keyword"
                },
                "question": {
                    "type": "text",
                    "analyzer": analyzer_name,
                    "search_analyzer": search_analyzer_name
                },
                "answer": {
                    "type": "text",
                    "analyzer": analyzer_name,
                    "search_analyzer": search_analyzer_name
                }
            }
        }
    }
    
    # 创建索引
    try:
        es_client.indices.create(
            index=index_name,
            settings=index_mapping["settings"],
            mappings=index_mapping["mappings"]
        )
        print(f"  ✓ 成功创建混合索引: {index_name}")
        print(f"  向量维度: {vector_dimension}")
        print(f"  相似度算法: cosine（余弦相似度）")
        print(f"  分词器: {analyzer_name}")
        print(f"  支持功能: 向量搜索（KNN）+ 文本搜索（BM25）+ 混合搜索（Hybrid Search）")
    except Exception as e:
        # 如果IK分词器不可用，尝试使用标准分词器
        error_msg = str(e)
        if "ik" in error_msg.lower() or "analyzer" in error_msg.lower():
            print(f"  ⚠ IK分词器不可用，使用标准分词器")
            analyzer_name = "standard"
            search_analyzer_name = "standard"
            index_mapping["mappings"]["properties"]["title"]["analyzer"] = analyzer_name
            index_mapping["mappings"]["properties"]["title"]["search_analyzer"] = search_analyzer_name
            index_mapping["mappings"]["properties"]["content"]["analyzer"] = analyzer_name
            index_mapping["mappings"]["properties"]["content"]["search_analyzer"] = search_analyzer_name
            index_mapping["mappings"]["properties"]["question"]["analyzer"] = analyzer_name
            index_mapping["mappings"]["properties"]["question"]["search_analyzer"] = search_analyzer_name
            index_mapping["mappings"]["properties"]["answer"]["analyzer"] = analyzer_name
            index_mapping["mappings"]["properties"]["answer"]["search_analyzer"] = search_analyzer_name
            
            es_client.indices.create(
                index=index_name,
                settings=index_mapping["settings"],
                mappings=index_mapping["mappings"]
            )
            print(f"  ✓ 成功创建混合索引: {index_name}（使用标准分词器）")
            print(f"  支持功能: 向量搜索（KNN）+ 文本搜索（BM25）+ 混合搜索（Hybrid Search）")
        else:
            print(f"  ✗ 创建索引失败: {error_msg}")
            raise


def index_documents_text(es_client: Elasticsearch, index_name: str, chunks: List[Dict]):
    """
    将文档块索引到Elasticsearch（非向量模式，只保存文本数据）
    
    参数:
        es_client: Elasticsearch客户端对象
        index_name: 索引名称
        chunks: 文档块列表
    
    功能:
        - 使用bulk API批量索引文档
        - 为每个文档块生成唯一ID
        - 显示索引进度
    """
    if not chunks:
        print("⚠ 没有文档块需要索引")
        return
    
    # 准备批量索引的数据
    actions = []
    for i, chunk in enumerate(chunks):
        # 生成文档ID
        doc_id = f"{hashlib.sha256(chunk['source'].encode()).hexdigest()[:16]}_{chunk['chunk_id']}"
        
        # 构建要索引的文档（不包含向量字段）
        action = {
            "_index": index_name,
            "_id": doc_id,
            "_source": chunk
        }
        actions.append(action)
    
    print(f"📤 准备索引 {len(actions)} 个文档块（非向量模式）...")
    
    try:
        # 使用bulk API批量索引
        success_count, failed_items = bulk(es_client, actions, chunk_size=BATCH_SIZE, request_timeout=60)
        
        print(f"✓ 索引完成！")
        print(f"  成功索引: {success_count} 个文档")
        if failed_items:
            print(f"  失败: {len(failed_items)} 个文档")
            for item in failed_items[:5]:  # 只显示前5个失败项
                print(f"    - {item}")
        
        # 刷新索引，使新索引的文档立即可搜索
        es_client.indices.refresh(index=index_name)
        print(f"  索引已刷新，文档可立即搜索")
        
    except Exception as e:
        print(f"✗ 索引失败: {str(e)}")
        raise


def index_documents_with_vectors(es_client: Elasticsearch, index_name: str, chunks: List[Dict], 
                                  embeddings: np.ndarray):
    """
    将文档块和向量索引到Elasticsearch（混合索引，同时包含向量和文本字段）
    
    参数:
        es_client: Elasticsearch客户端对象
        index_name: 索引名称
        chunks: 文档块列表
        embeddings: 向量矩阵（numpy数组）
    
    功能:
        - 为每个文档块生成向量嵌入
        - 使用bulk API批量索引文档和向量
        - 单个索引同时包含向量字段和文本字段，支持Hybrid Search
        - 显示索引进度
    """
    if not chunks:
        print("⚠ 没有文档块需要索引")
        return
    
    if embeddings is None or len(embeddings) == 0:
        print("⚠ 没有向量数据需要索引")
        return
    
    if len(chunks) != len(embeddings):
        raise Exception(f"文档块数量({len(chunks)})与向量数量({len(embeddings)})不匹配")
    
    # 准备批量索引的数据
    actions = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # 生成文档ID
        doc_id = f"{hashlib.sha256(chunk['source'].encode()).hexdigest()[:16]}_{chunk['chunk_id']}"
        
        # 构建要索引的文档（包含向量字段）
        doc_data = chunk.copy()
        doc_data['embedding'] = embedding.tolist()  # numpy数组转列表
        
        action = {
            "_index": index_name,
            "_id": doc_id,
            "_source": doc_data
        }
        actions.append(action)
    
    print(f"📤 准备索引 {len(actions)} 个文档块（包含向量）...")
    
    try:
        # 批量索引（向量数据较大，使用较小的chunk_size）
        success_count, failed_items = bulk(es_client, actions, chunk_size=50, request_timeout=120)
        
        print(f"✓ 索引完成！")
        print(f"  成功索引: {success_count} 个文档")
        if failed_items:
            print(f"  失败: {len(failed_items)} 个文档")
            for item in failed_items[:5]:  # 只显示前5个失败项
                print(f"    - {item}")
        
        # 刷新索引
        es_client.indices.refresh(index=index_name)
        print(f"  索引已刷新，文档可立即搜索")
        
    except Exception as e:
        print(f"✗ 索引失败: {str(e)}")
        raise


# ============================================================================
# QA对数据读取和转换
# ============================================================================

def prepare_chunks_from_qa_pairs(domain: str = 'system') -> List[Dict]:
    """
    从QA对JSON文件中读取数据并转换为文档块格式
    支持读取多个QA对文件（每个文档一个文件）
    
    参数:
        domain: 域类型（policy/system）
    
    返回:
        文档块列表，每个块包含title, source, content等信息
    
    功能:
        - 读取 data/qa_pairs/{domain}/*.json 所有文件
        - 将每个QA对转换为文档块格式
        - 使用question和answer组合作为content
        - 保留每个文档的metadata信息
    """
    # 构建QA对目录路径
    qa_dir = QA_PAIRS_DIR / domain
    
    if not qa_dir.exists():
        raise FileNotFoundError(f"QA对目录不存在: {qa_dir}")
    
    # 查找所有 *.json 文件
    qa_files = list(qa_dir.glob('*.json'))
    
    if not qa_files:
        raise FileNotFoundError(f"未找到JSON文件: {qa_dir}/*.json")
    
    print(f"📁 找到 {len(qa_files)} 个JSON文件")
    
    all_chunks = []
    total_qa_pairs = 0
    
    # 遍历所有QA对文件
    for qa_file in qa_files:
        print(f"  正在读取: {qa_file.name}")
        
        try:
            # 读取JSON文件
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            if not isinstance(qa_pairs, list):
                print(f"    ⚠ 跳过: 文件格式不正确（不是数组）")
                continue
            
            print(f"    ✓ 读取 {len(qa_pairs)} 个QA对")
            total_qa_pairs += len(qa_pairs)
            
            # 将QA对转换为文档块格式
            for idx, qa in enumerate(qa_pairs):
                # 提取QA对信息
                qa_id = qa.get('id', f'{domain}_{idx:04d}')
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                created_at = qa.get('created_at', '')
                
                # 从metadata中提取信息（如果存在）
                doc_id = qa.get('doc_id', f"{domain}_{qa_id}")
                doc_type = qa.get('doc_type', '操作手册' if domain == 'system' else '监管政策')
                region = qa.get('region', '全国')
                status = qa.get('status', '生效')
                role = qa.get('role', '客户经理')
                domain_value = qa.get('domain', '系统' if domain == 'system' else '政策')
                
                # 组合question和answer作为content（便于搜索）
                # 格式：问题：xxx\n答案：xxx
                content = f"问题：{question}\n答案：{answer}"
                
                # 计算token数量（简单估算：1 token ≈ 4字符）
                token_count = len(content) // 4
                
                # 构建文档块数据
                chunk = {
                    'title': question[:100] if len(question) > 100 else question,  # 使用问题作为标题
                    'source': str(qa_file),  # 源文件路径
                    'content': content,      # 问题和答案的组合内容
                    'chunk_id': len(all_chunks),  # 块ID（全局索引）
                    'page_num': 1,           # 页码（QA对通常不分页）
                    'token_count': token_count,  # Token数量
                    'file_type': 'json',     # 文件类型
                    'domain': domain_value,  # 域类型（从metadata获取）
                    'doc_type': doc_type,     # 文档类型（从metadata获取）
                    'region': region,        # 地区范围（从metadata获取）
                    'publish_date': created_at[:10] if created_at else '2024-01-01',  # 发布时间（提取日期部分）
                    'status': status,        # 状态（从metadata获取）
                    'role': role,            # 角色（从metadata获取）
                    'doc_id': doc_id,        # 文档ID（从metadata获取）
                    # 额外字段：保存原始QA对信息
                    'qa_id': qa_id,         # QA对ID
                    'question': question,    # 原始问题
                    'answer': answer        # 原始答案
                }
                all_chunks.append(chunk)
        
        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON解析失败: {e}")
            continue
        except Exception as e:
            print(f"    ⚠ 读取文件失败: {e}")
            continue
    
    print(f"\n✓ 成功读取 {len(qa_files)} 个文件，共 {total_qa_pairs} 个QA对")
    print(f"✓ 成功转换 {len(all_chunks)} 个文档块")
    
    if all_chunks:
        print(f"  示例：")
        print(f"    - 标题: {all_chunks[0]['title']}")
        print(f"    - 内容长度: {len(all_chunks[0]['content'])} 字符")
        print(f"    - Token数: {all_chunks[0]['token_count']}")
        print(f"    - 文档ID: {all_chunks[0]['doc_id']}")
    
    return all_chunks


# ============================================================================
# 主处理函数
# ============================================================================

def process_domain(domain: str, model, tokenizer):
    """
    处理某个域的QA对数据，生成向量库
    
    参数:
        domain: 域类型（policy/system）
        model: Embedding模型
        tokenizer: Tokenizer
    """
    print("=" * 60)
    print(f"处理 {domain.upper()} 域QA对数据")
    print("=" * 60)
    
    # 检查QA对目录是否存在
    qa_dir = QA_PAIRS_DIR / domain
    if not qa_dir.exists():
        print(f"⚠ 跳过: QA对目录不存在: {qa_dir}")
        return
    
    # 检查是否有JSON文件
    qa_files = list(qa_dir.glob('*.json'))
    if not qa_files:
        print(f"⚠ 跳过: 未找到JSON文件: {qa_dir}/*.json")
        return
    
    # 从QA对文件读取数据
    chunks = prepare_chunks_from_qa_pairs(domain)
    
    if not chunks:
        print("⚠ 没有可用的文档块")
        return
    
    print(f"\n总共生成 {len(chunks)} 个文档块")
    
    # 生成embeddings（使用配置的batch_size）
    texts = [chunk['content'] for chunk in chunks]
    embeddings = generate_embeddings(texts, model, tokenizer, batch_size=EMBEDDING_BATCH_SIZE)
    
    # 连接Elasticsearch
    try:
        es_client = Elasticsearch(**ES_CONFIG)
        # 再次测试连接（确保连接正常）
        if not es_client.ping():
            raise Exception("无法连接到Elasticsearch服务器")
    except Exception as e:
        print(f"  ⚠ Elasticsearch连接失败: {e}")
        raise
    
    # 获取索引配置
    index_config = INDEX_CONFIG.get(domain)
    if not index_config:
        print(f"  ⚠ 未找到域 {domain} 的索引配置")
        return
    
    index_name = index_config['index_name']
    vector_dimension = embeddings.shape[1]
    
    # 更新向量维度（如果与配置不一致）
    if vector_dimension != index_config['vector_dimension']:
        print(f"  ⚠ 向量维度 {vector_dimension} 与配置不一致，更新配置")
        index_config['vector_dimension'] = vector_dimension
    
    # 创建混合索引（同时支持向量搜索和文本搜索，用于ES原生Hybrid Search）
    print(f"\n正在创建混合索引（支持Hybrid Search）: {index_name}")
    create_vector_index(es_client, index_name, vector_dimension)
    
    # 索引文档和向量（单个索引包含向量和文本字段，支持混合搜索）
    print(f"\n正在索引数据（包含向量和文本字段）...")
    index_documents_with_vectors(es_client, index_name, chunks, embeddings)
    
    # 显示索引统计信息
    stats = es_client.count(index=index_name)
    print(f"✓ 混合索引 {index_name} 中共有 {stats['count']} 条文档")
    print(f"  - 支持向量搜索（KNN）")
    print(f"  - 支持文本搜索（BM25）")
    print(f"  - 支持混合搜索（Hybrid Search：BM25 + 向量）")
    
    print(f"\n✓ {domain} 域向量库生成完成！")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("RAG向量库生成脚本（基于QA对数据）")
    print("=" * 60)
    
    # 检查依赖
    if not NUMPY_AVAILABLE:
        print("⚠ 错误: numpy未安装")
        return
    
    if not MODELSCOPE_AVAILABLE:
        print("⚠ 错误: modelscope或transformers未安装")
        return
    
    if not ELASTICSEARCH_AVAILABLE:
        print("⚠ 错误: elasticsearch未安装")
        return
    
    # 先测试Elasticsearch连接（在加载模型之前，避免浪费资源）
    if not test_elasticsearch_connection():
        print("\n⚠ 错误: Elasticsearch连接失败，无法继续")
        print("   请先解决Elasticsearch连接问题，然后再运行脚本")
        return
    
    # 加载embedding模型（连接成功后再加载，避免浪费资源）
    print(f"\n正在加载embedding模型: {EMBEDDING_MODEL_NAME}...")
    try:
        model, tokenizer = load_embedding_model(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"⚠ 模型加载失败: {e}")
        return
    
    print()
    
    # 处理policy和system两个域
    domains = ['policy', 'system']
    
    for domain in domains:
        # 处理该域的QA对数据
        try:
            process_domain(domain, model, tokenizer)
        except Exception as e:
            print(f"⚠ 处理 {domain} 域失败: {e}")
            import traceback
            traceback.print_exc()
        
        if domain != domains[-1]:
            print("\n" + "="*60 + "\n")
    
    print("\n" + "=" * 60)
    print("向量库生成完成！")
    print("=" * 60)
    print("\n索引说明：")
    print(f"  政策类：")
    print(f"    - {INDEX_CONFIG['policy']['index_name']}: 混合索引（支持Hybrid Search）")
    print(f"  系统功能类：")
    print(f"    - {INDEX_CONFIG['system']['index_name']}: 混合索引（支持Hybrid Search）")
    print("\n查询示例：")
    print("  # 混合搜索（Hybrid Search：BM25 + 向量）")
    print("  POST /bank_credit_policy/_search")
    print("  # 向量搜索（KNN）")
    print("  POST /bank_credit_policy/_search")
    print("  # 文本搜索（BM25）")
    print("  POST /bank_credit_policy/_search")


if __name__ == "__main__":
    main()
