"""
RAG查询模块
功能：
1. Query改写（调用大模型）
2. 向量化查询
3. Elasticsearch向量搜索
4. 重排序（可配置）
5. 大模型生成最终答案
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass

# 添加项目根目录到路径（从src/rag/到项目根目录需要两级）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入配置
from config.elasticsearch import ES_CONFIG, INDEX_CONFIG
from config.prompts import get_rag_query_prompt, TODAY
from config.rag_config import RAG_CONFIG, get_rag_config, is_rerank_enabled, is_rewrite_enabled

# 导入监控模块
from src.utils.llm_monitor import set_token_info
from src.utils.monitor import extract_token_info_from_response

# Embedding模型配置（与build_vector_db.py保持一致）
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')

# 大模型配置（与generate_qa_pairs.py保持一致）
LLM_MODE = os.getenv('LLM_MODE', 'bailian').lower()
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
BAILIAN_MODEL = os.getenv('BAILIAN_MODEL', 'qwen-plus')
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')

# ============================================================================
# 依赖检查
# ============================================================================

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠ 警告: elasticsearch未安装")

try:
    from modelscope import snapshot_download
    from transformers import AutoTokenizer, AutoModel
    import torch
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("⚠ 警告: modelscope或transformers未安装")

try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============================================================================
# 全局变量（模型缓存）
# ============================================================================

_embedding_model = None
_embedding_tokenizer = None
_local_llm_model = None
_local_llm_tokenizer = None

# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class SearchResult:
    """搜索结果数据类"""
    content: str              # 文档内容
    score: float             # 相似度分数
    metadata: Dict           # 元数据
    chunk_id: str            # 文本块ID

# ============================================================================
# Embedding模型加载和向量化
# ============================================================================

def load_embedding_model(model_path: str = None):
    """
    加载Embedding模型（全局缓存）
    
    参数:
        model_path: 模型路径，如果为None，使用全局配置
    """
    global _embedding_model, _embedding_tokenizer
    
    if _embedding_model is not None:
        return _embedding_model, _embedding_tokenizer
    
    if not MODELSCOPE_AVAILABLE:
        raise Exception("modelscope或transformers未安装")
    
    if model_path is None:
        model_path = EMBEDDING_MODEL_NAME
    
    print(f"  正在加载Embedding模型: {model_path}")
    
    try:
        # 下载模型（如果未下载）
        model_dir = snapshot_download(model_path, cache_dir='./models/embedding')
        
        # 加载tokenizer和model
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 检查是否有GPU
        use_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        if use_cpu:
            device_map = 'cpu'
        elif torch.cuda.is_available():
            device_map = 'auto'
        else:
            device_map = 'cpu'
        
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, device_map=device_map)
        model.eval()
        
        _embedding_model = model
        _embedding_tokenizer = tokenizer
        
        print(f"  ✓ Embedding模型加载成功")
        return model, tokenizer
        
    except Exception as e:
        raise Exception(f"Embedding模型加载失败: {e}")


def generate_query_embedding(query: str) -> np.ndarray:
    """
    生成查询文本的embedding向量
    
    参数:
        query: 查询文本
    
    返回:
        np.ndarray: embedding向量
    """
    model, tokenizer = load_embedding_model()
    
    # Tokenize
    inputs = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # 移动到模型设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # 提取embedding（与build_vector_db.py保持一致）
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output.cpu().numpy()[0]
        elif hasattr(outputs, 'last_hidden_state'):
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        else:
            embedding = outputs.cpu().numpy()[0] if isinstance(outputs, torch.Tensor) else outputs[0]
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.numpy()
    
    return embedding

# ============================================================================
# 大模型调用（复用generate_qa_pairs.py的逻辑）
# ============================================================================

def load_local_llm_model(model_path: str = None):
    """加载本地LLM模型（全局缓存）"""
    global _local_llm_model, _local_llm_tokenizer
    
    if _local_llm_model is not None:
        return _local_llm_model, _local_llm_tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        raise Exception("transformers未安装")
    
    if model_path is None:
        model_path = LOCAL_MODEL_PATH
    
    print(f"  正在加载本地LLM模型: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        use_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
        if use_cpu:
            device_map = 'cpu'
        elif torch.cuda.is_available():
            device_map = 'auto'
        else:
            device_map = 'cpu'
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.float16 if device_map != 'cpu' else torch.float32
        )
        model.eval()
        
        _local_llm_model = model
        _local_llm_tokenizer = tokenizer
        
        print(f"  ✓ 本地LLM模型加载成功")
        return model, tokenizer
        
    except Exception as e:
        raise Exception(f"本地LLM模型加载失败: {e}")


def call_local_llm(prompt: str, max_length: int = 1000, module: str = "rag_query") -> str:
    """
    调用本地LLM模型
    
    参数:
        prompt: 提示词
        max_length: 最大生成长度
        module: 模块名称（已废弃，保留以兼容旧代码）
    
    返回:
        str: 模型生成的文本
    """
    model, tokenizer = load_local_llm_model()
    
    # 构建提示词（Qwen2.5格式）
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize（获取实际token数）
    inputs = tokenizer.encode(formatted_prompt, return_tensors='pt')
    prompt_tokens = inputs.shape[1]  # 获取实际token数
    
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant的回复
    if '<|im_start|>assistant\n' in generated_text:
        response = generated_text.split('<|im_start|>assistant\n')[-1]
        response = response.split('<|im_end|>')[0].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()
    
    # 计算completion tokens（生成的总token数减去输入的token数）
    # outputs.shape[1] 是完整序列长度（输入+生成），prompt_tokens 是输入长度
    # 差值就是生成的token数（使用max(0, ...)防止负数）
    completion_tokens = max(0, outputs.shape[1] - prompt_tokens)
    
    # 设置token信息到线程本地存储（供监控装饰器使用）
    set_token_info(prompt_tokens, completion_tokens)
    
    return response


def call_bailian_api(prompt: str, module: str = "rag_query") -> str:
    """
    调用百炼API
    
    参数:
        prompt: 提示词
        module: 模块名称（已废弃，保留以兼容旧代码）
    
    返回:
        str: 模型生成的文本
    """
    if not DASHSCOPE_AVAILABLE:
        raise Exception("dashscope未安装")
    
    if not DASHSCOPE_API_KEY:
        raise ValueError("百炼API密钥未设置，请设置 DASHSCOPE_API_KEY 环境变量")
    
    dashscope.api_key = DASHSCOPE_API_KEY
    
    response = Generation.call(
        model=BAILIAN_MODEL,
        prompt=prompt,
        temperature=0.7,
        max_tokens=1000,
        result_format='message'
    )
    
    if response.status_code == 200:
        # 提取token信息（从响应对象中）
        prompt_tokens = 0
        completion_tokens = 0
        try:
            token_info = extract_token_info_from_response(response, BAILIAN_MODEL)
            prompt_tokens = token_info.get('prompt_tokens', 0)
            completion_tokens = token_info.get('completion_tokens', 0)
        except Exception:
            # 如果提取失败，使用默认值0
            pass
        
        content = None
        if 'output' in response:
            output = response['output']
            if 'choices' in output and len(output['choices']) > 0:
                message = output['choices'][0].get('message', {})
                if 'content' in message:
                    content = message['content'].strip()
            if not content and 'text' in output:
                content = output['text'].strip()
        
        if not content and 'text' in response:
            content = response['text'].strip()
        if not content and 'content' in response:
            content = response['content'].strip()
        
        if content:
            # 设置token信息到线程本地存储（供监控装饰器使用）
            set_token_info(prompt_tokens, completion_tokens)
            return content
    
    error_msg = f"百炼API调用失败: {response.status_code}"
    if hasattr(response, 'message'):
        error_msg += f" - {response.message}"
    raise Exception(error_msg)


def call_bailian_api_stream(prompt: str, module: str = "rag_query") -> Generator[str, None, None]:
    """
    调用百炼API（流式输出）
    
    参数:
        prompt: 提示词
        module: 模块名称（已废弃，保留以兼容旧代码）
    
    返回:
        Generator[str, None, None]: 生成器，逐步返回模型生成的文本片段
    """
    if not DASHSCOPE_AVAILABLE:
        raise Exception("dashscope未安装")
    
    if not DASHSCOPE_API_KEY:
        raise ValueError("百炼API密钥未设置，请设置 DASHSCOPE_API_KEY 环境变量")
    
    dashscope.api_key = DASHSCOPE_API_KEY
    
    # 使用流式调用
    responses = Generation.call(
        model=BAILIAN_MODEL,
        prompt=prompt,
        temperature=0.7,
        max_tokens=1000,
        result_format='message',
        stream=True  # 启用流式输出
    )
    
    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        chunk_count = 0
        previous_content = ""  # 用于计算增量内容
        
        for response in responses:
            if response.status_code == 200:
                # 提取内容
                # 百炼API流式响应格式：
                # - delta.content: 增量内容（推荐使用）
                # - message.content: 累积的完整内容（需要计算增量）
                if 'output' in response:
                    output = response['output']
                    if 'choices' in output and len(output['choices']) > 0:
                        choice = output['choices'][0]
                        delta = choice.get('delta', {})
                        message = choice.get('message', {})
                        
                        # 优先从delta获取增量内容
                        chunk_text = None
                        if 'content' in delta:
                            # delta.content 是增量内容，直接使用
                            chunk_text = delta['content']
                        elif 'content' in message:
                            # message.content 是累积内容，需要计算增量
                            current_content = message['content']
                            if current_content.startswith(previous_content):
                                # 计算增量部分
                                chunk_text = current_content[len(previous_content):]
                                previous_content = current_content
                            else:
                                # 如果内容不连续，可能是新的响应，直接使用
                                chunk_text = current_content
                                previous_content = current_content
                        
                        if chunk_text:
                            full_content += chunk_text
                            chunk_count += 1
                            if chunk_count <= 3:  # 只打印前3个chunk的日志
                                print(f"[call_bailian_api_stream] 收到chunk #{chunk_count} (长度: {len(chunk_text)}): {chunk_text[:50]}...")
                            yield chunk_text
                        else:
                            # 检查是否有finish_reason（表示流式结束）
                            finish_reason = choice.get('finish_reason')
                            if finish_reason:
                                print(f"[call_bailian_api_stream] 流式结束，finish_reason: {finish_reason}")
                else:
                    # 检查响应结构
                    if chunk_count == 0:
                        print(f"[call_bailian_api_stream] 响应中没有output字段，响应结构: {list(response.keys())}")
                
                # 提取token信息（从最后一个响应中）
                try:
                    token_info = extract_token_info_from_response(response, BAILIAN_MODEL)
                    prompt_tokens = token_info.get('prompt_tokens', 0)
                    completion_tokens = token_info.get('completion_tokens', 0)
                except Exception:
                    pass
            else:
                error_msg = f"百炼API流式调用失败: {response.status_code}"
                if hasattr(response, 'message'):
                    error_msg += f" - {response.message}"
                print(f"[call_bailian_api_stream] API错误: {error_msg}")
                raise Exception(error_msg)
        
        print(f"[call_bailian_api_stream] 流式调用完成，总共收到 {chunk_count} 个chunk，完整内容长度: {len(full_content)}")
        
        # 设置token信息到线程本地存储（供监控装饰器使用）
        set_token_info(prompt_tokens, completion_tokens)
        
    except Exception as e:
        # 如果流式调用失败，抛出异常
        error_msg = f"百炼API流式调用异常: {e}"
        print(f"[call_bailian_api_stream] 异常: {error_msg}")
        import traceback
        traceback.print_exc()
        raise Exception(error_msg)


def call_llm(prompt: str, mode: str = None, module: str = "rag_query", stream: bool = False) -> str:
    """
    调用大模型（统一接口，带监控）
    
    参数:
        prompt: 提示词
        mode: 调用模式（'bailian' 或 'local'），如果为None，使用全局配置
        module: 模块名称（用于监控）
        stream: 是否使用流式输出（默认False）
    
    返回:
        str: 模型生成的文本（如果stream=True，返回完整文本）
    """
    if mode is None:
        mode = LLM_MODE
    
    if stream:
        # 流式输出（目前只支持百炼API）
        if mode == 'local':
            # 本地模型暂不支持流式输出，降级为普通调用
            return call_local_llm(prompt, module=module)
        else:
            # 百炼API流式输出，收集所有片段后返回完整文本
            full_text = ""
            for chunk in call_bailian_api_stream(prompt, module=module):
                full_text += chunk
            return full_text
    else:
        # 普通输出
        if mode == 'local':
            return call_local_llm(prompt, module=module)
        else:
            return call_bailian_api(prompt, module=module)


def call_llm_stream(prompt: str, mode: str = None, module: str = "rag_query") -> Generator[str, None, None]:
    """
    调用大模型（流式输出，返回生成器）
    
    参数:
        prompt: 提示词
        mode: 调用模式（'bailian' 或 'local'），如果为None，使用全局配置
        module: 模块名称（用于监控）
    
    返回:
        Generator[str, None, None]: 生成器，逐步返回模型生成的文本片段
    """
    if mode is None:
        mode = LLM_MODE
    
    if mode == 'local':
        # 本地模型暂不支持流式输出，降级为普通调用后逐字符返回
        full_text = call_local_llm(prompt, module=module)
        # 模拟流式输出（逐字符返回）
        for char in full_text:
            yield char
    else:
        # 百炼API流式输出
        yield from call_bailian_api_stream(prompt, module=module)

# ============================================================================
# Elasticsearch向量搜索
# ============================================================================

def search_vectors(
    query_vector: np.ndarray,
    index_name: str,
    domain: str = None,
    role: str = '客户经理',
    top_k: int = None,
    filters: Dict = None
) -> List[SearchResult]:
    """
    在Elasticsearch中搜索相似向量
    
    参数:
        query_vector: 查询向量
        index_name: 索引名称
        domain: 域类型（用于过滤）
        role: 用户角色（用于权限过滤）
        top_k: 返回的文档数量
        filters: 额外的过滤条件
    
    返回:
        List[SearchResult]: 搜索结果列表
    """
    if not ELASTICSEARCH_AVAILABLE:
        raise Exception("elasticsearch未安装")
    
    if top_k is None:
        config = get_rag_config()
        top_k = config['top_k']
    
    # 连接Elasticsearch
    es_client = Elasticsearch(**ES_CONFIG)
    
    # 构建查询
    query = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector.tolist(),
            "k": top_k,
            "num_candidates": max(50, top_k * 10),  # 推荐值：至少 50，或 top_k*10
            # "num_candidates": top_k * 2,  # 候选数量
        }
    }
    
    # 添加过滤条件
    must_filters = []
    
    # 权限过滤（根据用户角色）
    if role:
        must_filters.append({"term": {"role": role}})
    
    # 域过滤（使用英文值：policy/system）
    if domain:
        must_filters.append({"term": {"domain": domain}})
    
    # 状态过滤（只查询生效的文档）
    must_filters.append({"term": {"status": "生效"}})
    
    # 添加额外过滤条件
    if filters:
        for key, value in filters.items():
            must_filters.append({"term": {key: value}})
    
    # 如果有过滤条件，添加到查询中
    if must_filters:
        query["knn"]["filter"] = {
            "bool": {
                "must": must_filters
            }
        }
    
    # 调试：输出查询信息
    # print(f"  🔍 索引名称: {index_name}")
    # print(f"  🔍 过滤条件: {must_filters}")
    # print(f"  🔍 查询top_k: {top_k}")
    
    # 执行搜索
    try:
        # 先检查索引是否存在
        if not es_client.indices.exists(index=index_name):
            print(f"  ⚠ 索引 {index_name} 不存在！")
            return []
        
        # 检查索引中的文档总数
        index_stats = es_client.count(index=index_name)
        total_docs = index_stats['count']
        print(f"  📊 索引中的文档总数: {total_docs}")
        
        # 如果没有文档，直接返回
        if total_docs == 0:
            print(f"  ⚠ 索引 {index_name} 中没有文档！")
            return []
        
        response = es_client.search(index=index_name, body=query, size=top_k)
        
        # 调试：输出搜索结果统计
        total_hits = response['hits']['total']
        if isinstance(total_hits, dict):
            total_hits = total_hits.get('value', 0)
        print(f"  📊 匹配到的文档数: {total_hits}")
        
        results = []
        for hit in response['hits']['hits']:
            result = SearchResult(
                content=hit['_source'].get('content', ''),
                score=hit['_score'],
                metadata={
                    'domain': hit['_source'].get('domain', ''),
                    'doc_type': hit['_source'].get('doc_type', ''),
                    'source': hit['_source'].get('source', ''),
                    'region': hit['_source'].get('region', ''),
                    'publish_date': hit['_source'].get('publish_date', ''),
                    'doc_id': hit['_source'].get('doc_id', ''),
                },
                chunk_id=hit['_source'].get('chunk_id', hit['_id'])
            )
            results.append(result)
        
        # 如果没有结果，尝试不带过滤条件的搜索，查看索引中是否有数据
        if len(results) == 0 and must_filters:
            print(f"  ⚠ 带过滤条件未找到结果，尝试查看索引中的文档样本...")
            sample_query = {
                "size": 3,
                "query": {"match_all": {}}
            }
            try:
                sample_response = es_client.search(index=index_name, body=sample_query)
                if sample_response['hits']['hits']:
                    print(f"  📝 索引中的文档样本（前3条）:")
                    for i, hit in enumerate(sample_response['hits']['hits'], 1):
                        source = hit['_source']
                        print(f"    文档{i}:")
                        print(f"      domain: {source.get('domain', 'N/A')}")
                        print(f"      role: {source.get('role', 'N/A')}")
                        print(f"      status: {source.get('status', 'N/A')}")
                        print(f"      content: {source.get('content', '')[:100]}...")
            except Exception as e:
                print(f"  ⚠ 获取文档样本失败: {e}")
        
        return results
        
    except Exception as e:
        raise Exception(f"Elasticsearch搜索失败: {e}")

# ============================================================================
# 重排序
# ============================================================================

def rerank_results(
    query: str,
    results: List[SearchResult],
    top_k: int = None,
    method: str = None
) -> List[SearchResult]:
    """
    对搜索结果进行重排序
    
    参数:
        query: 原始查询
        results: 搜索结果列表
        top_k: 返回的文档数量
        method: 重排序方法（'similarity' 或 'bm25'）
    
    返回:
        List[SearchResult]: 重排序后的结果
    """
    config = get_rag_config()
    
    if top_k is None:
        top_k = config['rerank_top_k']
    
    if method is None:
        method = config['rerank_method']
    
    if not results:
        return []
    
    # 方法1：基于相似度分数排序（简单但有效）
    if method == 'similarity':
        # 按分数降序排序
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    
    # 方法2：BM25分数（需要实现BM25计算，这里简化处理）
    elif method == 'bm25':
        # 简化版：结合相似度分数和文本匹配度
        # 实际应用中可以使用更复杂的BM25算法
        for result in results:
            # 简单的文本匹配度计算
            query_words = set(query.lower().split())
            content_words = set(result.content.lower().split())
            match_ratio = len(query_words & content_words) / max(len(query_words), 1)
            # 综合分数 = 向量相似度 * 0.7 + 文本匹配度 * 0.3
            result.score = result.score * 0.7 + match_ratio * 0.3
        
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    
    else:
        # 默认：按相似度排序
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]

# ============================================================================
# 大模型生成最终答案
# ============================================================================

def generate_answer(
    query: str,
    search_results: List[SearchResult],
    domain: str = 'general',
    stream: bool = False
) -> str:
    """
    使用大模型基于检索结果生成最终答案
    
    参数:
        query: 用户查询
        search_results: 搜索结果列表
        domain: 域类型（'policy'/'system'/'general'）
        stream: 是否使用流式输出（默认False）
    
    返回:
        str: 生成的答案（如果stream=True，返回完整文本）
    """
    if not search_results:
        return "抱歉，未找到相关信息。请尝试使用其他关键词查询。"
    
    # 构建上下文（合并检索到的文档内容）
    context_parts = []
    for i, result in enumerate(search_results, 1):
        # 添加元数据信息
        metadata_info = []
        if result.metadata.get('source'):
            metadata_info.append(f"来源：{result.metadata['source']}")
        if result.metadata.get('publish_date'):
            metadata_info.append(f"发布时间：{result.metadata['publish_date']}")
        if result.metadata.get('region'):
            metadata_info.append(f"地区：{result.metadata['region']}")
        
        metadata_str = " | ".join(metadata_info) if metadata_info else ""
        
        context_part = f"[文档{i}]"
        if metadata_str:
            context_part += f" ({metadata_str})"
        context_part += f"\n{result.content}\n"
        context_parts.append(context_part)
    
    context = "\n".join(context_parts)
    
    # 获取提示词模板
    prompt_template = get_rag_query_prompt(domain)
    
    # 填充提示词（包含日期信息）
    prompt = prompt_template.format(context=context, question=query, today=TODAY)
    
    # 调用大模型生成答案
    try:
        answer = call_llm(prompt, module="rag_answer", stream=stream)
        return answer.strip()
    except Exception as e:
        return f"生成答案时出错: {e}"


def generate_answer_stream(
    query: str,
    search_results: List[SearchResult],
    domain: str = 'general'
) -> Generator[str, None, None]:
    """
    使用大模型基于检索结果生成最终答案（流式输出）
    
    参数:
        query: 用户查询
        search_results: 搜索结果列表
        domain: 域类型（'policy'/'system'/'general'）
    
    返回:
        Generator[str, None, None]: 生成器，逐步返回答案文本片段
    """
    if not search_results:
        yield "抱歉，未找到相关信息。请尝试使用其他关键词查询。"
        return
    
    # 构建上下文（合并检索到的文档内容）
    context_parts = []
    for i, result in enumerate(search_results, 1):
        # 添加元数据信息
        metadata_info = []
        if result.metadata.get('source'):
            metadata_info.append(f"来源：{result.metadata['source']}")
        if result.metadata.get('publish_date'):
            metadata_info.append(f"发布时间：{result.metadata['publish_date']}")
        if result.metadata.get('region'):
            metadata_info.append(f"地区：{result.metadata['region']}")
        
        metadata_str = " | ".join(metadata_info) if metadata_info else ""
        
        context_part = f"[文档{i}]"
        if metadata_str:
            context_part += f" ({metadata_str})"
        context_part += f"\n{result.content}\n"
        context_parts.append(context_part)
    
    context = "\n".join(context_parts)
    
    # 获取提示词模板
    prompt_template = get_rag_query_prompt(domain)
    
    # 填充提示词（包含日期信息）
    prompt = prompt_template.format(context=context, question=query, today=TODAY)
    
    # 调用大模型生成答案（流式输出）
    try:
        print(f"[generate_answer_stream] 开始调用流式LLM，prompt长度: {len(prompt)}")
        chunk_count = 0
        for chunk in call_llm_stream(prompt, module="rag_answer"):
            chunk_count += 1
            if chunk_count <= 3:  # 只打印前3个chunk的日志
                print(f"[generate_answer_stream] 收到chunk #{chunk_count}: {chunk[:50]}...")
            yield chunk
        print(f"[generate_answer_stream] 流式生成完成，总共收到 {chunk_count} 个chunk")
        if chunk_count == 0:
            print(f"[generate_answer_stream] ⚠️ 警告：没有收到任何chunk")
    except Exception as e:
        error_msg = f"生成答案时出错: {e}"
        print(f"[generate_answer_stream] 异常: {error_msg}")
        yield error_msg

# ============================================================================
# RAG查询主函数
# ============================================================================

def rag_query(
    query: str,
    domain: str = 'policy',
    role: str = '客户经理',
    enable_rewrite: bool = True,
    enable_rerank: bool = None,
    filters: Dict = None
) -> Dict:
    """
    RAG查询主函数
    
    参数:
        query: 用户查询
        domain: 域类型（'policy'/'system'）
        role: 用户角色（用于权限过滤）
        enable_rewrite: 是否启用query改写
        enable_rerank: 是否启用重排序（如果为None，使用配置中的默认值）
        filters: 额外的过滤条件
    
    返回:
        Dict: 包含答案和检索结果的字典
    """
    print(f"\n{'='*60}")
    print(f"RAG查询")
    print(f"{'='*60}")
    print(f"原始查询: {query}")
    print(f"域类型: {domain}")
    print(f"用户角色: {role}")
    
    # 步骤1：Query改写（如果启用且未在外部改写）
    if enable_rewrite:
        print(f"\n[步骤1] Query改写...")
        # 导入提示词
        from config.prompts import QUERY_REWRITE_PROMPT, TODAY
        # 构建提示词并调用LLM（包含日期信息）
        prompt = QUERY_REWRITE_PROMPT.format(original_query=query, today=TODAY)
        rewritten_query = call_llm(prompt, module="rag_query_rewrite")
        rewritten_query = rewritten_query.strip().strip('"').strip("'")
        if not rewritten_query or len(rewritten_query) < 3:
            rewritten_query = query
        print(f"  改写后: {rewritten_query}")
        search_query = rewritten_query
    else:
        # 如果已在外部改写（如app.py），直接使用传入的query
        search_query = query
        rewritten_query = None
    
    # 步骤2：向量化
    print(f"\n[步骤2] 向量化查询...")
    try:
        query_vector = generate_query_embedding(search_query)
        print(f"  ✓ 向量维度: {query_vector.shape[0]}")
    except Exception as e:
        raise Exception(f"向量化失败: {e}")
    
    # 步骤3：Elasticsearch向量搜索
    print(f"\n[步骤3] Elasticsearch向量搜索...")
    try:
        index_config = INDEX_CONFIG.get(domain)
        if not index_config:
            raise Exception(f"未找到域 {domain} 的索引配置")
        
        index_name = index_config['index_name']
        config = get_rag_config()
        search_results = search_vectors(
            query_vector=query_vector,
            index_name=index_name,
            domain=domain,
            role=role,
            top_k=config['top_k'],
            filters=filters
        )
        print(f"  ✓ 检索到 {len(search_results)} 条结果")
    except Exception as e:
        raise Exception(f"向量搜索失败: {e}")
    
    # 步骤4：重排序（可选）
    config = get_rag_config()
    
    if enable_rerank is None:
        enable_rerank = config['enable_rerank']
    
    if enable_rerank and search_results:
        print(f"\n[步骤4] 重排序...")
        search_results = rerank_results(
            query=search_query,
            results=search_results,
            top_k=config['rerank_top_k'],
            method=config['rerank_method']
        )
        print(f"  ✓ 重排序后保留 {len(search_results)} 条结果")
    
    # 过滤低分结果
    config = get_rag_config()
    filtered_results = [
        r for r in search_results 
        if r.score >= config['min_score']
    ]
    
    if not filtered_results:
        return {
            'answer': "抱歉，未找到相关信息。请尝试使用其他关键词查询。",
            'query': query,
            'rewritten_query': rewritten_query if enable_rewrite else None,
            'results': [],
            'domain': domain
        }
    
    # 步骤5：大模型生成最终答案
    print(f"\n[步骤5] 大模型生成答案...")
    # 检查是否启用流式输出（仅对最后一次LLM总结结果调用使用流式）
    enable_streaming = RAG_CONFIG.get('enable_streaming', True)
    answer = generate_answer(
        query=query,  # 使用原始查询，不是改写后的
        search_results=filtered_results,
        domain=domain,
        stream=enable_streaming
    )
    print(f"  ✓ 答案生成完成")
    
    return {
        'answer': answer,
        'query': query,
        'rewritten_query': rewritten_query if enable_rewrite else None,
        'results': [
            {
                'content': r.content,
                'score': r.score,
                'metadata': r.metadata,
                'chunk_id': r.chunk_id
            }
            for r in filtered_results
        ],
        'domain': domain
    }


# ============================================================================
# 主函数（用于测试）
# ============================================================================

def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG查询测试')
    parser.add_argument('query', type=str, help='查询问题')
    parser.add_argument('--domain', type=str, default='policy', choices=['policy', 'system'],
                       help='域类型（policy/system）')
    parser.add_argument('--role', type=str, default='客户经理',
                       help='用户角色（客户经理/团队负责人/行长）')
    parser.add_argument('--no-rewrite', action='store_true',
                       help='禁用query改写')
    parser.add_argument('--no-rerank', action='store_true',
                       help='禁用重排序')
    
    args = parser.parse_args()
    
    try:
        result = rag_query(
            query=args.query,
            domain=args.domain,
            role=args.role,
            enable_rewrite=not args.no_rewrite,
            enable_rerank=not args.no_rerank
        )
        
        print(f"\n{'='*60}")
        print(f"查询结果")
        print(f"{'='*60}")
        print(f"\n答案：\n{result['answer']}")
        
        if result['rewritten_query']:
            print(f"\n改写后的查询：{result['rewritten_query']}")
        
        print(f"\n检索到的文档数量：{len(result['results'])}")
        for i, r in enumerate(result['results'][:3], 1):  # 只显示前3条
            print(f"\n[文档{i}] 相似度: {r['score']:.4f}")
            print(f"  内容: {r['content'][:100]}...")
            print(f"  来源: {r['metadata'].get('source', '未知')}")
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

