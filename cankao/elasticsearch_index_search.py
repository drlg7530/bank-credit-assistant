"""
Elasticsearch 文档索引和搜索脚本

功能：
1. 连接到 Elasticsearch
2. 创建索引（如果不存在）
3. 解析 docs 文件夹中的文档并分块
4. 将文档块索引到 Elasticsearch
5. 执行搜索查询
6. 显示搜索结果

使用前准备：
1. 安装依赖：
   pip install elasticsearch -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install "qwen-agent[rag]" -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 如果 ES 不在本地，请修改 ES_HOST 配置

3. （可选）安装 IK 分词器插件以获得更好的中文分词效果：
   bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.x.x/elasticsearch-analysis-ik-8.x.x.zip
   注意：版本号需要与 ES 版本匹配

作者：AI Assistant
日期：2025-12-27
"""

import os
import json
from typing import List, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from qwen_agent.tools.doc_parser import DocParser
from qwen_agent.settings import DEFAULT_WORKSPACE

# ====== 步骤 1：配置 Elasticsearch 连接 ======
# Elasticsearch 连接配置
# 注意：如果 ES 不在本地，请修改 ES_HOST 为实际的服务器地址
ES_HOST = "localhost"  # ES 服务器地址，根据实际情况修改（如：192.168.1.100）
ES_PORT = 9200  # ES 端口，默认 9200
ES_USERNAME = "elastic"  # ES 用户名
ES_PASSWORD = "elastic"  # ES 密码（请根据实际情况修改）

# 索引配置
INDEX_NAME = "pingan_employer_insurance"  # 索引名称

# 文档文件夹路径
DOCS_FOLDER = "./docs"  # 文档文件夹路径

# 分词器配置
# 如果 ES 安装了 IK 分词器插件，使用 "ik_max_word"
# 如果没有安装 IK 分词器，将使用 "standard"（标准分词器）
USE_IK_ANALYZER = True  # 是否使用 IK 分词器（需要先安装 IK 插件）


# ====== 步骤 2：连接到 Elasticsearch ======
def connect_elasticsearch():
    """
    连接到 Elasticsearch 服务器
    
    功能：
    - 使用用户名和密码进行身份验证
    - 测试连接是否成功
    
    返回：
    - Elasticsearch 客户端对象
    """
    print("=" * 60)
    print("步骤 1：连接到 Elasticsearch")
    print("=" * 60)
    
    # 创建 ES 客户端
    # 使用 HTTP Basic Auth 进行身份验证
    # 注意：elasticsearch 9.x 版本使用 request_timeout 而不是 timeout
    # 对于 HTTPS 连接，需要禁用 SSL 证书验证（适用于自签名证书）
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # 禁用 SSL 警告
    
    # 先尝试 HTTPS 连接
    es_urls = [
        f"https://{ES_HOST}:{ES_PORT}",  # 优先尝试 HTTPS
        f"http://{ES_HOST}:{ES_PORT}"    # 备用 HTTP
    ]
    
    last_error = None
    for es_url in es_urls:
        try:
            print(f"正在尝试连接: {es_url}...")
            es = Elasticsearch(
                [es_url],
                basic_auth=(ES_USERNAME, ES_PASSWORD),  # 9.x 版本使用 basic_auth 而不是 http_auth
                request_timeout=30,  # 请求超时时间（秒）
                max_retries=2,  # 最大重试次数
                retry_on_timeout=True,  # 超时时重试
                verify_certs=False,  # 禁用 SSL 证书验证（适用于自签名证书）
                ssl_show_warn=False  # 不显示 SSL 警告
            )
            
            # 测试连接
            cluster_info = es.info()
            print(f"✅ 成功连接到 Elasticsearch: {es_url}")
            print(f"   集群名称: {cluster_info['cluster_name']}")
            print(f"   ES 版本: {cluster_info['version']['number']}")
            return es
        except Exception as e:
            last_error = e
            print(f"   ⚠️  {es_url} 连接失败: {str(e)[:100]}")
            continue
    
    # 如果所有连接都失败，抛出最后一个错误
    if last_error is None:
        last_error = Exception("无法连接到 Elasticsearch：所有连接方式都失败")
    
    e = last_error
    error_msg = str(e)
    error_type = type(e).__name__
    print(f"❌ 连接 Elasticsearch 失败")
    print(f"   错误类型: {error_type}")
    print(f"   错误信息: {error_msg}")
    print(f"\n尝试连接的地址: https://{ES_HOST}:{ES_PORT} 和 http://{ES_HOST}:{ES_PORT}")
    print(f"用户名: {ES_USERNAME}")
    print(f"密码: {'*' * len(ES_PASSWORD)} (已隐藏)")
    
    # 尝试提供更详细的错误信息和解决方案
    print("\n" + "=" * 60)
    print("诊断信息：")
    print("=" * 60)
    
    if "certificate" in error_msg.lower() or "ssl" in error_msg.lower():
        print("💡 SSL 证书相关错误")
        print("   已自动禁用证书验证，如果仍有问题，请检查 ES 的 SSL 配置")
    elif "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
        print("💡 认证失败")
        print("   请检查用户名和密码是否正确")
        print("   可以在浏览器中访问 https://localhost:9200 测试认证")
    elif "connection" in error_msg.lower() or "refused" in error_msg.lower() or "timeout" in error_msg.lower():
        print("💡 连接问题")
        print("   请确认：")
        print("   1. Elasticsearch 服务已启动")
        print("   2. 端口 9200 未被其他程序占用")
        print("   3. 防火墙未阻止连接")
    elif "NameResolutionError" in error_type or "DNS" in error_msg:
        print("💡 DNS 解析问题")
        print("   请检查主机地址是否正确")
    else:
        print("💡 未知错误")
        print("   请查看上面的错误信息")
    
    print("\n建议的排查步骤：")
    print("1. 在浏览器中访问 https://localhost:9200")
    print("   如果浏览器能访问，说明 ES 服务正常")
    print("2. 检查 ES 日志文件，查看是否有错误信息")
    print("3. 确认用户名和密码是否正确")
    print("4. 尝试使用 curl 命令测试连接：")
    print(f'   curl -u {ES_USERNAME}:{ES_PASSWORD} -k https://localhost:9200')
    
    raise


# ====== 步骤 3：检查 IK 分词器是否可用 ======
def check_ik_analyzer(es: Elasticsearch) -> bool:
    """
    检查 IK 分词器是否已安装
    
    参数：
    - es: Elasticsearch 客户端对象
    
    返回：
    - True: IK 分词器可用
    - False: IK 分词器不可用
    """
    try:
        # 尝试创建一个临时索引来测试 IK 分词器
        test_index = "_test_ik_analyzer"
        try:
            # 先删除测试索引（如果存在）
            if es.indices.exists(index=test_index):
                es.indices.delete(index=test_index)
            
            # 创建测试索引
            test_mapping = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "test_ik": {
                                "type": "ik_max_word"
                            }
                        }
                    }
                }
            }
            # elasticsearch 9.x 版本：直接传递 settings 和 mappings 参数
            es.indices.create(
                index=test_index,
                settings=test_mapping["settings"],
                mappings=test_mapping.get("mappings", {})
            )
            es.indices.delete(index=test_index)
            return True
        except Exception:
            return False
    except Exception:
        return False


# ====== 步骤 4：创建索引 ======
def create_index(es: Elasticsearch, index_name: str):
    """
    创建 Elasticsearch 索引（如果不存在）
    
    参数：
    - es: Elasticsearch 客户端对象
    - index_name: 索引名称
    
    功能：
    - 定义索引的映射（mapping）结构
    - 设置中文分词器（ik_max_word 或 standard）
    - 创建索引（如果已存在则跳过）
    """
    print("\n" + "=" * 60)
    print("步骤 2：创建索引")
    print("=" * 60)
    
    # 检查索引是否已存在
    if es.indices.exists(index=index_name):
        print(f"⚠️  索引 '{index_name}' 已存在，将使用现有索引")
        print("   如需重新创建，请先删除现有索引")
        return
    
    # 选择分词器
    # 如果配置使用 IK 分词器，先检查是否可用
    if USE_IK_ANALYZER:
        ik_available = check_ik_analyzer(es)
        if ik_available:
            analyzer_name = "ik_max_word"
            search_analyzer_name = "ik_max_word"
            print(f"   ✅ IK 分词器可用，使用: ik_max_word")
        else:
            analyzer_name = "standard"
            search_analyzer_name = "standard"
            print(f"   ⚠️  IK 分词器不可用，自动切换到标准分词器")
            print(f"   提示: 如需更好的中文分词效果，请安装 IK 分词器插件")
            print(f"   安装命令: bin/elasticsearch-plugin install analysis-ik")
    else:
        analyzer_name = "standard"
        search_analyzer_name = "standard"
        print(f"   使用分词器: 标准分词器 (standard)")
    
    index_mapping = {
        "settings": {
            # 设置分片和副本数
            "number_of_shards": 1,  # 主分片数
            "number_of_replicas": 0,  # 副本数（开发环境可以设为 0）
        },
        "mappings": {
            "properties": {
                # 文档标题字段
                "title": {
                    "type": "text",  # 文本类型，支持全文搜索
                    "analyzer": analyzer_name,  # 使用指定的分词器
                    "search_analyzer": search_analyzer_name,  # 搜索时也使用指定的分词器
                    "fields": {
                        "keyword": {
                            "type": "keyword"  # 保留原始值，用于精确匹配
                        }
                    }
                },
                # 文档源文件路径
                "source": {
                    "type": "keyword"  # 关键字类型，不进行分词
                },
                # 文档块内容（主要搜索字段）
                "content": {
                    "type": "text",
                    "analyzer": analyzer_name,
                    "search_analyzer": search_analyzer_name
                },
                # 块 ID（在同一文档中的序号）
                "chunk_id": {
                    "type": "integer"
                },
                # 页码（如果文档有分页）
                "page_num": {
                    "type": "integer"
                },
                # Token 数量
                "token_count": {
                    "type": "integer"
                },
                # 文档类型（txt, pdf, docx 等）
                "file_type": {
                    "type": "keyword"
                }
            }
        }
    }
    
    try:
        # 创建索引
        # elasticsearch 9.x 版本：直接传递 settings 和 mappings 参数
        es.indices.create(
            index=index_name,
            settings=index_mapping["settings"],
            mappings=index_mapping["mappings"]
        )
        print(f"✅ 成功创建索引: {index_name}")
        print(f"   索引映射已配置")
    except Exception as e:
        print(f"❌ 创建索引失败: {str(e)}")
        raise


# ====== 步骤 4：解析文档并分块 ======
def parse_and_chunk_documents(docs_folder: str) -> List[Dict]:
    """
    解析 docs 文件夹中的文档并分块
    
    参数：
    - docs_folder: 文档文件夹路径
    
    返回：
    - 文档块列表，每个块包含 title, source, content, chunk_id 等信息
    
    功能：
    - 遍历 docs 文件夹中的所有文件
    - 使用 DocParser 解析文档
    - 将文档分块（chunk）
    - 返回所有文档块
    """
    print("\n" + "=" * 60)
    print("步骤 3：解析文档并分块")
    print("=" * 60)
    
    # 创建 DocParser 实例
    # 配置分块大小：每个块最多 500 tokens
    doc_parser = DocParser({
        'max_ref_token': 20000,  # 最大参考 token 数
        'parser_page_size': 500  # 每个块的最大 token 数
    })
    
    # 获取所有文档文件
    if not os.path.exists(docs_folder):
        raise FileNotFoundError(f"文档文件夹不存在: {docs_folder}")
    
    # 获取文件夹中的所有文件
    files = []
    for file in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, file)
        if os.path.isfile(file_path):  # 确保是文件而不是目录
            files.append(file_path)
    
    print(f"📁 找到 {len(files)} 个文档文件")
    
    # 解析所有文档
    all_chunks = []
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] 正在解析: {os.path.basename(file_path)}")
        
        try:
            # 使用 DocParser 解析文档
            # 这会自动处理文档分块和缓存
            record = doc_parser.call(params={'url': file_path})
            
            # record 结构：
            # {
            #   'url': 文件路径,
            #   'title': 文档标题,
            #   'raw': [
            #     {
            #       'content': 块内容,
            #       'metadata': {'source': 源文件, 'title': 标题, 'chunk_id': 块ID},
            #       'token': token 数量
            #     },
            #     ...
            #   ]
            # }
            
            # 获取文件扩展名（文件类型）
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            # 处理每个文档块
            for chunk in record['raw']:
                chunk_data = {
                    'title': record['title'],  # 文档标题
                    'source': file_path,  # 源文件路径
                    'content': chunk['content'],  # 块内容
                    'chunk_id': chunk['metadata'].get('chunk_id', 0),  # 块 ID
                    'page_num': chunk['metadata'].get('page_num', 1),  # 页码（如果有）
                    'token_count': chunk['token'],  # Token 数量
                    'file_type': file_ext  # 文件类型
                }
                all_chunks.append(chunk_data)
            
            print(f"   ✅ 解析完成，生成 {len(record['raw'])} 个文档块")
            
        except Exception as e:
            print(f"   ❌ 解析失败: {str(e)}")
            print(f"   跳过此文件，继续处理其他文件")
            continue
    
    print(f"\n✅ 文档解析完成！")
    print(f"   总共生成 {len(all_chunks)} 个文档块")
    
    return all_chunks


# ====== 步骤 5：索引文档到 Elasticsearch ======
def index_documents(es: Elasticsearch, index_name: str, chunks: List[Dict]):
    """
    将文档块索引到 Elasticsearch
    
    参数：
    - es: Elasticsearch 客户端对象
    - index_name: 索引名称
    - chunks: 文档块列表
    
    功能：
    - 使用 bulk API 批量索引文档
    - 为每个文档块生成唯一 ID
    - 显示索引进度
    """
    print("\n" + "=" * 60)
    print("步骤 4：索引文档到 Elasticsearch")
    print("=" * 60)
    
    if not chunks:
        print("⚠️  没有文档块需要索引")
        return
    
    # 准备批量索引的数据
    actions = []
    for i, chunk in enumerate(chunks):
        # 生成文档 ID：使用源文件路径和块 ID 的组合
        doc_id = f"{hash(chunk['source'])}_{chunk['chunk_id']}"
        
        # 构建要索引的文档
        action = {
            "_index": index_name,  # 索引名称
            "_id": doc_id,  # 文档 ID
            "_source": chunk  # 文档内容
        }
        actions.append(action)
    
    print(f"📤 准备索引 {len(actions)} 个文档块...")
    
    try:
        # 使用 bulk API 批量索引
        # bulk 函数会自动处理批量操作，提高效率
        success_count, failed_items = bulk(es, actions, chunk_size=100, request_timeout=60)
        
        print(f"✅ 索引完成！")
        print(f"   成功索引: {success_count} 个文档")
        if failed_items:
            print(f"   失败: {len(failed_items)} 个文档")
            for item in failed_items[:5]:  # 只显示前 5 个失败项
                print(f"      - {item}")
        
        # 刷新索引，使新索引的文档立即可搜索
        es.indices.refresh(index=index_name)
        print(f"   索引已刷新，文档可立即搜索")
        
    except Exception as e:
        print(f"❌ 索引失败: {str(e)}")
        raise


# ====== 步骤 6：执行搜索 ======
def search_documents(es: Elasticsearch, index_name: str, search_query: str, top_k: int = 10):
    """
    在 Elasticsearch 中搜索文档
    
    参数：
    - es: Elasticsearch 客户端对象
    - index_name: 索引名称
    - search_query: 搜索查询字符串
    - top_k: 返回前 K 个结果，默认 10
    
    返回：
    - 搜索结果列表
    
    功能：
    - 使用多字段搜索（title 和 content）
    - 使用中文分词器进行搜索
    - 返回相关性评分最高的文档
    """
    print("\n" + "=" * 60)
    print("步骤 5：执行搜索")
    print("=" * 60)
    
    print(f"🔍 搜索查询: {search_query}")
    print(f"   返回前 {top_k} 个结果")
    
    # 构建搜索查询
    # 使用 multi_match 查询，在 title 和 content 字段中搜索
    search_body = {
        "query": {
            "multi_match": {
                "query": search_query,  # 搜索查询
                "fields": ["title^2", "content"],  # 搜索字段，title 权重为 2（更重要）
                "type": "best_fields"  # 最佳字段匹配
            }
        },
        "highlight": {  # 高亮显示匹配的文本
            "fields": {
                "title": {},
                "content": {
                    "fragment_size": 200,  # 片段大小
                    "number_of_fragments": 3  # 返回的片段数
                }
            },
            "pre_tags": ["<mark>"],  # 高亮开始标签
            "post_tags": ["</mark>"]  # 高亮结束标签
        },
        "size": top_k  # 返回结果数量
    }
    
    try:
        # 执行搜索
        # elasticsearch 9.x 版本：直接传递 query、highlight 和 size 参数
        response = es.search(
            index=index_name,
            query=search_body["query"],
            highlight=search_body["highlight"],
            size=search_body["size"]
        )
        
        # 解析搜索结果
        hits = response['hits']['hits']
        total = response['hits']['total']['value']
        
        print(f"\n✅ 搜索完成！")
        print(f"   找到 {total} 个相关文档")
        print(f"   返回前 {len(hits)} 个结果\n")
        
        return hits
        
    except Exception as e:
        print(f"❌ 搜索失败: {str(e)}")
        raise


# ====== 步骤 7：显示搜索结果 ======
def display_search_results(hits: List[Dict]):
    """
    格式化并显示搜索结果
    
    参数：
    - hits: 搜索结果列表
    
    功能：
    - 显示每个结果的标题、来源、相关性评分
    - 显示高亮的内容片段
    - 格式化输出，便于阅读
    """
    print("=" * 60)
    print("步骤 6：显示搜索结果")
    print("=" * 60)
    
    if not hits:
        print("⚠️  没有找到相关文档")
        return
    
    # 遍历显示每个搜索结果
    for i, hit in enumerate(hits, 1):
        score = hit['_score']  # 相关性评分
        source = hit['_source']  # 文档内容
        highlight = hit.get('highlight', {})  # 高亮片段
        
        print(f"\n{'=' * 60}")
        print(f"结果 {i} (相关性评分: {score:.4f})")
        print(f"{'=' * 60}")
        
        # 显示标题（如果有高亮，显示高亮版本）
        title = highlight.get('title', [source.get('title', '无标题')])[0]
        print(f"📄 标题: {title}")
        
        # 显示来源文件
        source_file = os.path.basename(source.get('source', '未知'))
        print(f"📁 来源: {source_file}")
        
        # 显示文件类型和块信息
        print(f"📊 信息: 文件类型={source.get('file_type', '未知')}, "
              f"块ID={source.get('chunk_id', 0)}, "
              f"Token数={source.get('token_count', 0)}")
        
        # 显示高亮的内容片段
        if 'content' in highlight:
            print(f"\n💡 相关内容片段:")
            for fragment in highlight['content'][:3]:  # 最多显示 3 个片段
                # 移除 HTML 标签以便在终端显示（实际应用中可以保留）
                fragment_text = fragment.replace('<mark>', '【').replace('</mark>', '】')
                print(f"   {fragment_text}")
        else:
            # 如果没有高亮，显示原始内容的前 300 个字符
            content = source.get('content', '')
            preview = content[:300] + ('...' if len(content) > 300 else '')
            print(f"\n💡 内容预览:")
            print(f"   {preview}")
        
        print()
    
    print("=" * 60)


# ====== 主程序 ======
def main():
    """
    主程序：执行完整的索引和搜索流程
    
    流程：
    1. 连接到 Elasticsearch
    2. 创建索引（如果不存在）
    3. 解析文档并分块
    4. 索引文档到 Elasticsearch
    5. 执行搜索
    6. 显示搜索结果
    """
    print("\n" + "🚀" * 30)
    print("Elasticsearch 文档索引和搜索系统")
    print("🚀" * 30 + "\n")
    
    try:
        # 步骤 1：连接到 Elasticsearch
        es = connect_elasticsearch()
        
        # 步骤 2：创建索引
        create_index(es, INDEX_NAME)
        
        # 步骤 3：解析文档并分块
        chunks = parse_and_chunk_documents(DOCS_FOLDER)
        
        # 步骤 4：索引文档
        if chunks:
            index_documents(es, INDEX_NAME, chunks)
        else:
            print("⚠️  没有文档块需要索引，跳过索引步骤")
        
        # 步骤 5：执行搜索
        search_query = "工伤保险和雇主险有什么区别？"
        hits = search_documents(es, INDEX_NAME, search_query, top_k=10)
        
        # 步骤 6：显示搜索结果
        display_search_results(hits)
        
        print("\n" + "✅" * 30)
        print("所有步骤执行完成！")
        print("✅" * 30 + "\n")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# ====== 程序入口 ======
if __name__ == '__main__':
    exit(main())

