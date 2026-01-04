"""
从解析后的文档生成QA对脚本

功能：
1. 读取解析后的文档（data/parsed/）
2. 按条款/章节/语义块切分（不使用LLM）
3. 使用LLM生成QA对（Q为大模型生成、A为政策或操作手册的功能点）
4. 保存QA对到data/qa_pairs/目录

根据需求文档第5点：
- 对解析后文档初步切分并根据域生成Q、A对
- 1.按条款/章节/语义块切分，不使用llm
- 2.使用llm生成QA对，Q为大模型生成、A为政策或操作手册的功能点

提示词使用：
- QA_GENERATION_PROMPT_POLICY（政策类）
- QA_GENERATION_PROMPT_SYSTEM（系统功能类）
"""

import os
import json
import re
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入提示词配置
from config.prompts import (
    QA_GENERATION_PROMPT_POLICY,
    QA_GENERATION_PROMPT_SYSTEM,
    TODAY
)

# ============================================================================
# 大模型配置区域
# ============================================================================

# 大模型模式配置
# 可选值: 'bailian' (百炼API) 或 'local' (本地模型)
LLM_MODE = os.getenv('LLM_MODE', 'bailian').lower()  # 默认使用百炼API

# 百炼API模式配置
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')  # 百炼API密钥（必需）
BAILIAN_MODEL = os.getenv('BAILIAN_MODEL', 'qwen-plus')  # 百炼模型名称

# 本地模型模式配置
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')

# 答案长度限制配置
# 限制答案文本的最大长度，避免token过多导致API调用失败或成本过高
ANSWER_MAX_LENGTH = int(os.getenv('ANSWER_MAX_LENGTH', '1000'))  # 默认1000字符

# ============================================================================
# 依赖检查
# ============================================================================

# 百炼（DashScope）SDK相关
try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("⚠ 提示: dashscope未安装，将无法使用百炼API")
    print("   请安装: pip install dashscope -i https://pypi.tuna.tsinghua.edu.cn/simple")

# 本地模型相关
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ 提示: transformers未安装，将无法使用本地模型模式")

# ============================================================================
# 文本切分函数（不使用LLM）
# ============================================================================

def split_text_by_clauses(content: str, domain: str = 'policy') -> List[str]:
    """
    按条款/章节/语义块切分文档内容（不使用LLM）
    
    参数:
        content: 文档内容（Markdown格式）
        domain: 域类型（policy/system），用于调整切分策略
    
    返回:
        List[str]: 切分后的文本块列表（每个块作为一个功能点/答案）
    
    功能:
        - 识别条款标识（第一条、第二条等）
        - 识别章节标识（第一章、第二章等）
        - 识别语义块（段落、操作步骤等）
        - 过滤元数据和无意义内容
        - 不使用LLM，纯规则切分
    """
    clauses = []
    
    # 过滤元数据和无意义内容的模式
    metadata_patterns = [
        r'^字号大中小',
        r'^文章来源：',
        r'^打印本页',
        r'^关闭窗口',
        r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # 日期时间
        r'^$',  # 空行
    ]
    
    # 按行分割内容
    lines = content.split('\n')
    current_clause = []
    
    for line in lines:
        line = line.strip()
        
        # 跳过空行和元数据行
        if not line:
            # 空行时，如果当前有累积的条款，保存它
            if current_clause:
                clause_text = '\n'.join(current_clause).strip()
                if len(clause_text) > 20:  # 只保留有意义的条款（长度>20）
                    clauses.append(clause_text)
                current_clause = []
            continue
        
        # 检查是否是元数据行
        is_metadata = False
        for pattern in metadata_patterns:
            if re.match(pattern, line):
                is_metadata = True
                break
        
        if is_metadata:
            continue
        
        # 匹配条款标识（条款级别）
        clause_patterns = [
            r'^第[一二三四五六七八九十百]+条',  # 第一条、第二条等
            r'^第[一二三四五六七八九十]+章',  # 第一章、第二章等
            r'^（[一二三四五六七八九十]+）',  # （一）、（二）等
            r'^\([一二三四五六七八九十]+\)',  # (一)、(二)等
            r'^\d+[、.]',  # 1、 2. 等数字编号
        ]
        
        # 对于系统功能文档，也识别操作步骤
        if domain == 'system':
            clause_patterns.extend([
                r'^步骤\d+[：:]',  # 步骤1：、步骤2：等
                r'^场景\s*[A-Z]',  # 场景A、场景B等
                r'^功能\d+[：:]',  # 功能1：、功能2：等
            ])
        
        is_clause_start = False
        for pattern in clause_patterns:
            if re.match(pattern, line):
                is_clause_start = True
                break
        
        # 对于系统文档，也识别Markdown标题（## 级别以上）
        if domain == 'system' and line.startswith('##'):
            is_clause_start = True
        
        if is_clause_start:
            # 如果当前有累积的条款，先保存
            if current_clause:
                clause_text = '\n'.join(current_clause).strip()
                if len(clause_text) > 20:
                    clauses.append(clause_text)
                current_clause = []
            
            # 开始新的条款
            current_clause.append(line)
        else:
            # 继续累积当前条款
            if current_clause:
                current_clause.append(line)
            # 对于系统文档，也识别独立的操作说明段落
            elif domain == 'system' and len(line) > 30 and not line.startswith('#'):
                # 可能是独立的功能说明段落
                if len(line) > 50:  # 较长的段落可能是独立功能点
                    clauses.append(line)
            # 对于政策文档，如果没有条款标识，也识别有意义的段落
            elif domain == 'policy' and len(line) > 30:
                # 如果是有实际内容的段落（不是元数据），也作为独立功能点
                # 检查是否包含实际业务内容（包含中文、数字、业务关键词）
                if re.search(r'[\u4e00-\u9fa5]', line) and len(line) > 50:
                    # 如果当前没有累积的条款，且这行是有意义的，作为新条款开始
                    if not current_clause:
                        current_clause.append(line)
    
    # 保存最后一个条款
    if current_clause:
        clause_text = '\n'.join(current_clause).strip()
        if len(clause_text) > 20:
            clauses.append(clause_text)
    
    # 如果仍然没有切分到任何内容，尝试按段落切分
    if not clauses:
        # 将整个文档按空行分割成段落
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for para in paragraphs:
            # 过滤元数据段落
            is_metadata_para = False
            for pattern in metadata_patterns:
                if re.match(pattern, para):
                    is_metadata_para = True
                    break
            
            if not is_metadata_para and len(para) > 30:
                # 检查是否包含实际业务内容
                if re.search(r'[\u4e00-\u9fa5]', para):
                    clauses.append(para)
    
    # 去重（保留顺序）
    seen = set()
    unique_clauses = []
    for clause in clauses:
        # 使用前50个字符作为去重依据
        clause_key = clause[:50]
        if clause_key not in seen:
            seen.add(clause_key)
            unique_clauses.append(clause)
    
    return unique_clauses


# ============================================================================
# LLM调用函数
# ============================================================================

# 全局变量：本地模型实例（避免重复加载）
_local_model = None
_local_tokenizer = None


def load_local_model(model_path: str = None, device: str = None):
    """
    加载本地大模型
    
    参数:
        model_path: 模型路径或模型名称
        device: 设备（'cuda' 或 'cpu'），自动检测
    """
    global _local_model, _local_tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("请先安装transformers: pip install transformers torch")
    
    if _local_model is not None:
        return _local_model, _local_tokenizer
    
    if model_path is None:
        model_path = LOCAL_MODEL_PATH
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"  正在加载本地模型: {model_path} (设备: {device})...")
    
    try:
        _local_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None
        )
        
        if device == 'cpu':
            _local_model = _local_model.to(device)
        
        _local_model.eval()
        print(f"  ✓ 本地模型加载成功")
        
        return _local_model, _local_tokenizer
    
    except Exception as e:
        raise Exception(f"加载本地模型失败: {e}")


def call_local_llm(prompt: str, model_path: str = None, max_length: int = 1000) -> str:
    """
    使用本地模型生成文本
    
    参数:
        prompt: 提示词
        model_path: 模型路径（如果为None，使用全局配置）
        max_length: 最大生成长度
    
    返回:
        str: 生成的文本
    """
    global _local_model, _local_tokenizer
    
    # 加载模型（如果未加载）
    if _local_model is None:
        load_local_model(model_path)
    
    try:
        # 构建提示词（Qwen2.5格式）
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = _local_tokenizer.encode(formatted_prompt, return_tensors='pt')
        device = next(_local_model.parameters()).device
        inputs = inputs.to(device)
        
        # 生成
        with torch.no_grad():
            outputs = _local_model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=_local_tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = _local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取assistant的回复
        if '<|im_start|>assistant\n' in generated_text:
            response = generated_text.split('<|im_start|>assistant\n')[-1]
            response = response.split('<|im_end|>')[0].strip()
        else:
            response = generated_text[len(formatted_prompt):].strip()
        
        return response
    
    except Exception as e:
        raise Exception(f"本地模型生成失败: {e}")


def call_bailian_api(prompt: str, api_key: str = None, model: str = None) -> str:
    """
    调用百炼（DashScope）API生成文本
    
    参数:
        prompt: 提示词
        api_key: API密钥（如果为None，使用全局配置）
        model: 模型名称（如果为None，使用全局配置）
    
    返回:
        str: 生成的文本
    """
    if not DASHSCOPE_AVAILABLE:
        raise ImportError("请先安装dashscope: pip install dashscope")
    
    if api_key is None:
        api_key = DASHSCOPE_API_KEY
    
    if model is None:
        model = BAILIAN_MODEL
    
    if not api_key:
        raise ValueError("百炼API密钥未设置，请设置 DASHSCOPE_API_KEY 环境变量")
    
    try:
        # 设置API密钥
        dashscope.api_key = api_key
        
        # 调用百炼API
        response = Generation.call(
            model=model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
            result_format='message'
        )
        
        # 检查响应状态
        if response.status_code == 200:
            if 'output' in response:
                output = response['output']
                # message格式
                if 'choices' in output and len(output['choices']) > 0:
                    message = output['choices'][0].get('message', {})
                    if 'content' in message:
                        return message['content'].strip()
                # 兼容text格式
                if 'text' in output:
                    return output['text'].strip()
            
            # 兼容其他格式
            if 'text' in response:
                return response['text'].strip()
            if 'content' in response:
                return response['content'].strip()
        
        # 错误处理
        error_msg = f"百炼API调用失败: {response.status_code}"
        if hasattr(response, 'message'):
            error_msg += f" - {response.message}"
        elif 'message' in response:
            error_msg += f" - {response['message']}"
        raise Exception(error_msg)
    
    except Exception as e:
        raise Exception(f"百炼API请求失败: {e}")


def parse_qa_from_llm_response(response: str, domain: str) -> List[Dict]:
    """
    从LLM响应中解析QA对
    
    参数:
        response: LLM生成的响应文本
        domain: 域类型（policy/system）
    
    返回:
        List[Dict]: QA对列表
    """
    qa_pairs = []
    
    try:
        # 尝试解析JSON数组
        # 移除可能的markdown代码块标记
        response = response.strip()
        if response.startswith('```'):
            # 移除代码块标记
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
            response = response.strip()
        
        # 尝试找到JSON数组
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed_data = json.loads(json_str)
            
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    if isinstance(item, dict) and 'question' in item:
                        qa_pairs.append(item)
        else:
            # 如果不是JSON格式，尝试提取问题
            # 查找"question"字段
            lines = response.split('\n')
            current_qa = {}
            for line in lines:
                if '"question"' in line or "'question'" in line:
                    # 提取问题文本
                    match = re.search(r'["\']question["\']\s*[:：]\s*["\']([^"\']+)["\']', line)
                    if match:
                        current_qa['question'] = match.group(1)
                elif current_qa and ('question' in current_qa):
                    # 如果已经有问题，尝试提取其他字段
                    if '"doc_id"' in line or "'doc_id'" in line:
                        match = re.search(r'["\']doc_id["\']\s*[:：]\s*["\']([^"\']+)["\']', line)
                        if match:
                            current_qa['doc_id'] = match.group(1)
                    
                    if current_qa.get('question'):
                        qa_pairs.append(current_qa)
                        current_qa = {}
    
    except json.JSONDecodeError as e:
        print(f"    ⚠ JSON解析失败: {e}")
        # 如果JSON解析失败，尝试简单提取
        # 查找问题模式
        question_pattern = r'["\']question["\']\s*[:：]\s*["\']([^"\']+)["\']'
        matches = re.findall(question_pattern, response)
        for match in matches:
            qa_pairs.append({'question': match})
    
    return qa_pairs


def generate_qa_from_answer(answer: str, domain: str, doc_id: str = None) -> List[Dict]:
    """
    根据答案（功能点）生成QA对
    
    参数:
        answer: 答案文本（政策条款或系统功能点）
        domain: 域类型（policy/system）
        doc_id: 文档编号（可选）
    
    返回:
        List[Dict]: QA对列表（可能包含多个QA对）
    """
    # 根据域类型选择提示词
    if domain == 'policy':
        prompt_template = QA_GENERATION_PROMPT_POLICY
    else:
        prompt_template = QA_GENERATION_PROMPT_SYSTEM
    
    # 填充提示词模板
    # 限制答案长度，避免token过多（使用配置中的长度限制）
    answer_text = answer[:ANSWER_MAX_LENGTH] if len(answer) > ANSWER_MAX_LENGTH else answer
    
    # 根据提示词模板的占位符填充
    # 注意：提示词模板使用 {answer} 和 {today} 作为占位符
    prompt = prompt_template.format(answer=answer_text, today=TODAY)
    
    # 如果提示词中有其他占位符（如doc_id），需要额外处理
    # 但当前提示词模板中没有doc_id占位符，所以不需要处理
    
    try:
        # 调用LLM生成
        if LLM_MODE == 'local':
            response = call_local_llm(prompt)
        else:
            response = call_bailian_api(prompt)
        
        # 解析响应，提取QA对
        qa_pairs = parse_qa_from_llm_response(response, domain)
        
        # 为每个QA对添加答案
        for qa in qa_pairs:
            qa['answer'] = answer
            if 'doc_id' not in qa and doc_id:
                qa['doc_id'] = doc_id
        
        return qa_pairs
    
    except Exception as e:
        print(f"    ⚠ 生成QA对失败: {e}")
        # 如果生成失败，返回一个简单的QA对
        return [{
            'question': f"关于{answer[:30]}...的问题",
            'answer': answer,
            'doc_id': doc_id or 'unknown'
        }]


# ============================================================================
# 主处理函数
# ============================================================================

def extract_metadata_from_filename(filename: str, domain: str) -> Dict:
    """
    从文件名提取metadata
    
    参数:
        filename: 文件名
        domain: 域类型
    
    返回:
        Dict: metadata信息
    """
    metadata = {
        'doc_id': filename.replace('.md', '').replace('MinerU_markdown_', ''),
        'domain': domain,
        'doc_type': '监管政策' if domain == 'policy' else '操作手册',
        'source': filename,
        'region': '全国',  # 默认值，可以从文件名中提取
        'role': '客户经理',  # 默认值
        'status': '生效' # 默认值
    }
    
    return metadata


def save_qa_pairs_for_document(qa_pairs: List[Dict], md_file: Path, output_dir: Path, domain: str):
    """
    保存单个文档的QA对到文件
    
    参数:
        qa_pairs: QA对列表
        md_file: 原始markdown文件路径
        output_dir: 输出目录
        domain: 域类型（policy/system）
    """
    if not qa_pairs:
        return
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用文档的stem（不含扩展名）作为文件名
    # 清理文件名，移除特殊字符，避免文件系统问题
    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', md_file.stem)
    
    # 保存JSON格式
    json_file = output_dir / f"{safe_filename}_qa_pairs.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已保存 {len(qa_pairs)} 个QA对到: {json_file.name}")
    
    # 同时保存为CSV格式（便于查看）
    try:
        import pandas as pd
        csv_file = output_dir / f"{safe_filename}_qa_pairs.csv"
        df = pd.DataFrame(qa_pairs)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ 已保存CSV格式到: {csv_file.name}")
    except ImportError:
        print("  (跳过CSV保存，pandas未安装)")


def process_domain_documents(domain: str, parsed_dir: Path, output_dir: Path):
    """
    处理某个域的所有文档，生成QA对
    每个markdown文件生成的QA对保存为单独的文件
    
    参数:
        domain: 域类型（policy/system）
        parsed_dir: 解析后的文档目录
        output_dir: 输出目录
    """
    print("=" * 60)
    print(f"处理 {domain.upper()} 域文档")
    print("=" * 60)
    
    # 获取所有markdown文档
    md_files = list(parsed_dir.glob('*.md'))
    
    if not md_files:
        print(f"⚠ 未找到 {domain} 域的文档")
        return
    
    print(f"找到 {len(md_files)} 个文档文件")
    
    # 统计信息
    total_qa_pairs = 0
    processed_files = 0
    
    # 处理每个文档
    for md_file in md_files:
        print(f"\n处理文档: {md_file.name}")
        try:
            # 读取文档内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取metadata
            metadata = extract_metadata_from_filename(md_file.name, domain)
            
            # 按条款/章节/语义块切分（不使用LLM）
            print(f"  正在切分文档...")
            clauses = split_text_by_clauses(content, domain=domain)
            print(f"  切分得到 {len(clauses)} 个功能点")
            
            # 为当前文档收集QA对
            doc_qa_pairs = []
            
            # 为每个功能点生成QA对
            print(f"  正在生成QA对...")
            for idx, clause in enumerate(clauses, 1):
                print(f"    [{idx}/{len(clauses)}] 生成QA对...", end="", flush=True)
                
                try:
                    # 调用LLM生成QA对
                    qa_pairs = generate_qa_from_answer(
                        answer=clause,
                        domain=domain,
                        doc_id=metadata['doc_id']
                    )
                    
                    # 为每个QA对添加metadata和ID
                    for qa_idx, qa in enumerate(qa_pairs):
                        qa_id = f"{domain}_{md_file.stem}_{idx}_{qa_idx}"
                        qa['id'] = qa_id
                        qa['domain'] = domain
                        qa['created_at'] = datetime.now().isoformat()
                        # 添加metadata
                        qa.update(metadata)
                    
                    doc_qa_pairs.extend(qa_pairs)
                    print(f" ✓ (生成 {len(qa_pairs)} 个QA对)")
                
                except Exception as e:
                    print(f" ✗ ({e})")
                    continue
            
            # 保存当前文档的QA对到单独文件
            if doc_qa_pairs:
                save_qa_pairs_for_document(doc_qa_pairs, md_file, output_dir, domain)
                total_qa_pairs += len(doc_qa_pairs)
                processed_files += 1
            else:
                print(f"  ⚠ 该文档未生成任何QA对")
        
        except Exception as e:
            print(f"  ⚠ 处理文档失败: {e}")
            continue
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print(f"处理完成！")
    print(f"  处理文档数: {processed_files}/{len(md_files)}")
    print(f"  生成QA对总数: {total_qa_pairs}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


def main():
    """
    主函数
    """
    print("=" * 60)
    print("从解析后的文档生成QA对脚本")
    print("=" * 60)
    print(f"模式: {LLM_MODE.upper()}")
    
    # 检查依赖
    if LLM_MODE == 'local':
        if not TRANSFORMERS_AVAILABLE:
            print("⚠ 错误: transformers未安装，无法使用本地模型模式")
            print("   请安装: pip install transformers torch")
            return
    else:
        if not DASHSCOPE_AVAILABLE:
            print("⚠ 错误: dashscope未安装，无法使用百炼API")
            print("   请安装: pip install dashscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
            return
        if not DASHSCOPE_API_KEY:
            print("⚠ 警告: 百炼API密钥未设置")
            print("   请设置: export DASHSCOPE_API_KEY=your-api-key")
    
    print()
    
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    
    # 处理policy和system两个域
    domains = ['policy', 'system']
    
    for domain in domains:
        parsed_dir = project_root / "data" / "parsed" / domain
        output_dir = project_root / "data" / "qa_pairs" / domain
        
        if not parsed_dir.exists():
            print(f"⚠ 跳过: 解析目录不存在: {parsed_dir}")
            continue
        
        # 处理该域的文档
        process_domain_documents(domain, parsed_dir, output_dir)
        
        if domain != domains[-1]:
            print("\n" + "="*60 + "\n")
    
    print("\n" + "=" * 60)
    print("QA对生成完成！")
    print("=" * 60)
    
    print("\n配置说明：")
    print("  方式1: 修改脚本开头的配置变量（直接编辑文件）")
    print("  方式2: 使用环境变量（推荐）")
    print("\n【百炼API模式】（默认）")
    print("    export LLM_MODE=bailian")
    print("    export DASHSCOPE_API_KEY=your-api-key")
    print("    export BAILIAN_MODEL=qwen-plus")
    print("\n【本地模型模式】")
    print("    export LLM_MODE=local")
    print("    export LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct")


if __name__ == "__main__":
    main()

