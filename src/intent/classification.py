"""
问题理解与能力路由模块
功能：
1. 判断用户输入是否包含多个语义上独立的问题（问题拆分）
2. 使用大模型进行意图识别（CoT思维链推理）
3. 解析意图类型并选择对应的系统能力
4. 路由决策（RAG或预测模块）
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入配置
from config.prompts import INTENT_CLASSIFICATION_PROMPT, TODAY

# 导入监控模块
from src.utils.llm_monitor import llm_monitor, set_token_info
from src.utils.monitor import extract_token_info_from_response

# ============================================================================
# 配置区域
# ============================================================================

# 大模型配置（与rag_query.py保持一致）
LLM_MODE = os.getenv('LLM_MODE', 'bailian').lower()
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
BAILIAN_MODEL = os.getenv('BAILIAN_MODEL', 'qwen-plus')
LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')

# ============================================================================
# 依赖检查
# ============================================================================

try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============================================================================
# 意图类型枚举
# ============================================================================

class IntentType(Enum):
    """意图类型枚举"""
    POLICY_QUERY = "policy_query"           # 政策查询
    SYSTEM_QUERY = "system_query"           # 系统操作
    CUSTOMER_ANALYSIS = "customer_analysis" # 客户分析
    GENERAL = "general"                      # 一般性问题

# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class IntentResult:
    """意图识别结果"""
    intent: IntentType              # 意图类型
    confidence: float               # 置信度（0-1）
    reasoning: str                  # 推理过程（CoT输出）
    route_to: str                   # 路由目标（'rag_policy'/'rag_system'/'prediction'/'general'）
    # 实体信息（用于L2记忆）
    active_domain: List[str] = None  # 业务域列表（如：['system'], ['policy']等）
    business_object: str = ""       # 业务对象（如：押品、客户等）
    operation_stage: str = ""       # 操作阶段（如：创建、入库、审批等）

# ============================================================================
# 全局变量（模型缓存）
# ============================================================================

_local_llm_model = None
_local_llm_tokenizer = None

# ============================================================================
# 大模型调用（复用rag_query.py的逻辑）
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


def call_local_llm(prompt: str, max_length: int = 500, module: str = "intent_classification") -> str:
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


def call_bailian_api(prompt: str, module: str = "intent_classification") -> str:
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
        max_tokens=500,
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


def call_llm(prompt: str, mode: str = None, module: str = "intent_classification") -> str:
    """
    调用大模型（统一接口，带监控）
    
    参数:
        prompt: 提示词
        mode: 调用模式（'bailian' 或 'local'），如果为None，使用全局配置
        module: 模块名称（用于监控）
    
    返回:
        str: 模型生成的文本
    """
    if mode is None:
        mode = LLM_MODE
    
    if mode == 'local':
        return call_local_llm(prompt, module=module)
    else:
        return call_bailian_api(prompt, module=module)

# ============================================================================
# 意图识别核心函数
# ============================================================================

def parse_intent_and_entities_from_response(response: str) -> Tuple[IntentType, str, Dict]:
    """
    从模型响应中解析意图类型和实体信息（支持JSON格式和文本格式）
    
    参数:
        response: 模型响应文本（可能是JSON数组或包含CoT推理过程的文本）
    
    返回:
        Tuple[IntentType, str, Dict]: (意图类型, 推理过程, 实体信息字典)
        实体信息字典包含：active_domain, business_object, operation_stage
    """
    # 默认实体信息（不再包含active_domain，由规则映射）
    default_entities = {
        'business_object': '',
        'operation_stage': ''
    }
    
    # 调试：首先输出原始响应
    print(f"  🔍 [parse_intent_and_entities_from_response] 开始解析，响应长度: {len(response) if response else 0}")
    if response:
        print(f"  🔍 [parse_intent_and_entities_from_response] 响应内容（前500字符）: {response[:500]}...")
    else:
        print(f"  ⚠ [parse_intent_and_entities_from_response] 响应为空！")
        return IntentType.GENERAL, "", default_entities
    
    # 清理响应文本
    original_response = response
    response = response.strip()
    
    # 方法1：尝试解析JSON格式（根据提示词，应该返回JSON数组）
    try:
        # 尝试提取JSON部分（可能包含在代码块或其他文本中）
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # 尝试清理可能的代码块标记
            json_str = json_str.strip()
            if json_str.startswith('```'):
                # 移除代码块标记
                lines = json_str.split('\n')
                json_str = '\n'.join([line for line in lines if not line.strip().startswith('```')])
            json_str = json_str.strip()
            
            # 尝试解析JSON
            try:
                parsed_data = json.loads(json_str)
                print(f"  ✓ JSON解析成功")
            except json.JSONDecodeError as json_err:
                print(f"  ⚠ JSON解析失败: {json_err}")
                raise
            
            # 确保是数组格式
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                # 获取第一个元素的完整信息
                first_item = parsed_data[0]
                if isinstance(first_item, dict) and 'intent' in first_item:
                    intent_str = first_item['intent'].lower()
                    
                    # 映射到IntentType
                    intent_mapping = {
                        'policy_query': IntentType.POLICY_QUERY,
                        'system_query': IntentType.SYSTEM_QUERY,
                        'customer_analysis': IntentType.CUSTOMER_ANALYSIS,
                        'general': IntentType.GENERAL
                    }
                    
                    intent_type = intent_mapping.get(intent_str, IntentType.GENERAL)
                    
                    # 提取实体信息（不再提取active_domain，由规则映射）
                    entities = {
                        'business_object': first_item.get('business_object', ''),
                        'operation_stage': first_item.get('operation_stage', '')
                    }
                    
                    return intent_type, original_response, entities
                else:
                    print(f"  ⚠ JSON格式错误：第一个元素不是字典或缺少'intent'字段")
    except Exception as e:
        print(f"  ⚠ JSON解析失败（方法1）: {e}，尝试其他解析方法")
    
    # 方法2：尝试直接解析整个响应为JSON
    try:
        parsed_data = json.loads(response)
        if isinstance(parsed_data, list) and len(parsed_data) > 0:
            first_item = parsed_data[0]
            if isinstance(first_item, dict) and 'intent' in first_item:
                intent_str = first_item['intent'].lower()
                intent_mapping = {
                    'policy_query': IntentType.POLICY_QUERY,
                    'system_query': IntentType.SYSTEM_QUERY,
                    'customer_analysis': IntentType.CUSTOMER_ANALYSIS,
                    'general': IntentType.GENERAL
                }
                intent_type = intent_mapping.get(intent_str, IntentType.GENERAL)
                
                # 提取实体信息（不再提取active_domain，由规则映射）
                entities = {
                    'business_object': first_item.get('business_object', ''),
                    'operation_stage': first_item.get('operation_stage', '')
                }
                
                return intent_type, original_response, entities
    except Exception as e:
        print(f"  ⚠ JSON解析失败（方法2）: {e}")
    
    # 方法3：使用正则表达式匹配（降级方案，只解析意图，实体信息使用默认值）
    intent_patterns = {
        IntentType.POLICY_QUERY: [
            r'policy_query',
            r'政策查询',
            r'政策类',
            r'监管要求',
            r'政策规定'
        ],
        IntentType.SYSTEM_QUERY: [
            r'system_query',
            r'系统操作',
            r'系统类',
            r'如何操作',
            r'功能使用'
        ],
        IntentType.CUSTOMER_ANALYSIS: [
            r'customer_analysis',
            r'客户分析',
            r'客户风险',
            r'贷款意向',
            r'客户评估'
        ],
        IntentType.GENERAL: [
            r'general',
            r'一般性',
            r'通用'
        ]
    }
    
    # 查找匹配的意图类型
    for intent_type, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return intent_type, response, default_entities
    
    # 方法4：默认返回general
    return IntentType.GENERAL, response, default_entities


def parse_intent_from_response(response: str) -> Tuple[IntentType, str]:
    """
    从模型响应中解析意图类型（支持JSON格式和文本格式）
    兼容旧版本，调用新函数并只返回意图和推理过程
    
    参数:
        response: 模型响应文本（可能是JSON数组或包含CoT推理过程的文本）
    
    返回:
        Tuple[IntentType, str]: (意图类型, 推理过程)
    
    注意：此方法已废弃，现在使用 parse_intent_and_entities_from_response
    保留此方法仅用于向后兼容
    """
    intent, reasoning, _ = parse_intent_and_entities_from_response(response)
    return intent, reasoning


@llm_monitor(module="intent_classification")
def classify_intent(question: str, use_cot: bool = True) -> IntentResult:
    """
    问题理解与意图识别（支持CoT思维链推理）
    
    功能：
    1. 判断用户输入是否包含多个语义上独立的问题
    2. 如果只有一个清晰问题，保持原问题不拆分
    3. 只有在用户明确提出多个不同目标时，才拆分为多个子问题
    4. 为每个问题判断其意图类型，并选择对应的系统能力
    
    参数:
        question: 用户问题
        use_cot: 是否使用CoT思维链推理（默认True）
    
    返回:
        IntentResult: 意图识别结果（包含问题拆分信息）
    """
    # 构建提示词（包含日期信息）
    prompt = INTENT_CLASSIFICATION_PROMPT.format(question=question, today=TODAY)
    
    try:
        # 调用大模型
        print(f"  📞 正在调用LLM进行意图识别...")
        response = call_llm(prompt)
        print(f"  ✓ LLM调用完成，响应长度: {len(response) if response else 0}")
        
        # 调试：输出LLM的原始响应（仅前500字符，避免输出过长）
        if response:
            if len(response) > 500:
                print(f"  📝 LLM原始响应（前500字符）: {response[:500]}...")
            else:
                print(f"  📝 LLM原始响应: {response}")
        else:
            print(f"  ⚠ LLM返回的响应为空！")
            raise ValueError("LLM返回的响应为空")
        
        # 解析意图类型和实体信息
        print(f"  🔍 开始解析意图类型和实体信息...")
        intent, reasoning, entities = parse_intent_and_entities_from_response(response)
        print(f"  ✓ 意图解析完成，意图类型: {intent.value}")
        print(f"  ✓ 实体信息: {entities}")
        
        # 计算置信度（简化版：基于响应中是否明确包含意图类型）
        confidence = 0.8 if intent != IntentType.GENERAL else 0.5
        
        # 确定路由目标
        route_mapping = {
            IntentType.POLICY_QUERY: 'rag_policy',
            IntentType.SYSTEM_QUERY: 'rag_system',
            IntentType.CUSTOMER_ANALYSIS: 'prediction',
            IntentType.GENERAL: 'general'
        }
        route_to = route_mapping.get(intent, 'general')
        
        # 根据intent类型直接映射active_domain（规则映射，不需要LLM抽取）
        # 这样可以提高准确性和一致性，减少LLM工作量
        domain_mapping = {
            IntentType.POLICY_QUERY: ['policy'],
            IntentType.SYSTEM_QUERY: ['system'],
            IntentType.CUSTOMER_ANALYSIS: ['risk'],
            IntentType.GENERAL: []
        }
        active_domain = domain_mapping.get(intent, [])
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            route_to=route_to,
            active_domain=active_domain,  # 使用规则映射，而不是LLM抽取
            business_object=entities.get('business_object', ''),
            operation_stage=entities.get('operation_stage', '')
        )
        
    except Exception as e:
        # 改进错误处理：输出更详细的错误信息
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"  ⚠ 问题理解与意图识别失败: {error_type}: {error_msg}")
        
        # 输出完整的异常信息（包括所有属性）
        print(f"  ⚠ 异常类型: {error_type}")
        print(f"  ⚠ 异常消息: {error_msg}")
        print(f"  ⚠ 异常参数: {getattr(e, 'args', 'N/A')}")
        
        # 如果是JSON解析相关的错误，输出更多调试信息
        if 'sub_question' in error_msg or 'JSON' in error_msg or 'json' in error_msg or 'JSONDecodeError' in error_type:
            print(f"  ⚠ 可能是JSON解析错误，错误详情: {error_msg}")
            print(f"  ⚠ 用户问题: {question}")
            import traceback
            print(f"  ⚠ 完整错误堆栈:")
            traceback.print_exc()
        
        # 降级处理：使用简单的关键词匹配
        return fallback_intent_classification(question)


def fallback_intent_classification(question: str) -> IntentResult:
    """
    降级处理：基于关键词的简单意图识别
    
    参数:
        question: 用户问题
    
    返回:
        IntentResult: 意图识别结果
    """
    question_lower = question.lower()
    
    # 政策查询关键词（移除通用业务术语，只保留明确指向政策的关键词）
    # 注意：移除了"授信"、"信贷"、"银行"等通用术语，因为这些词在系统操作问题中也会出现
    policy_keywords = [
        '政策', '监管', '规定', '要求', '条款', '考核', '标准', '合规',
        '注册资本', '资本要求', 
        '监管要求', '政策规定', '制度', '办法', '通知', '意见', '指引'
    ]
    
    # 系统操作关键词（需要明确包含"系统"或"如何操作"等）
    # 注意：增加了"如何在"、"怎么在"等常见系统操作问法
    system_keywords = [
        '系统', '如何操作', '怎么操作', '操作步骤', '操作流程', '功能使用',
        '如何使用', '怎么使用', '系统功能', '系统查询', '系统申请',
        '如何在', '怎么在', '如何查询', '怎么查询', '如何申请', '怎么申请'
    ]
    
    # 客户分析关键词
    customer_keywords = [
        '风险', '预测', '分析', '评估', '意向', '趋势', '贷款需求',
        '客户风险', '风险评估', '客户分析', '客户评估'
    ]
    
    # 计算匹配分数（权重平均，所有类别权重相同）
    # 注意：所有关键词权重统一为1，确保公平匹配
    policy_score = sum(1 if kw in question_lower else 0 for kw in policy_keywords)
    system_score = sum(1 if kw in question_lower else 0 for kw in system_keywords)
    customer_score = sum(1 if kw in question_lower else 0 for kw in customer_keywords)
    
    # 判断意图类型（权重平均后的判断逻辑：按得分高低判断，得分相同时按优先级）
    # 优先级：系统查询 > 政策查询 > 客户分析 > 通用
    max_score = max(policy_score, system_score, customer_score)
    
    if max_score == 0:
        # 没有匹配到任何关键词，返回通用类型
        intent = IntentType.GENERAL
        route_to = 'general'
        confidence = 0.4
    elif system_score == max_score:
        # 系统得分最高，判断为系统查询
        intent = IntentType.SYSTEM_QUERY
        route_to = 'rag_system'
        confidence = 0.6
    elif policy_score == max_score:
        # 政策得分最高，判断为政策查询
        intent = IntentType.POLICY_QUERY
        route_to = 'rag_policy'
        confidence = 0.6
    elif customer_score == max_score:
        # 客户分析得分最高，判断为客户分析
        intent = IntentType.CUSTOMER_ANALYSIS
        route_to = 'prediction'
        confidence = 0.6
    else:
        # 兜底：返回通用类型
        intent = IntentType.GENERAL
        route_to = 'general'
        confidence = 0.4
    
    # 根据intent类型直接映射active_domain（规则映射）
    domain_mapping = {
        IntentType.POLICY_QUERY: ['policy'],
        IntentType.SYSTEM_QUERY: ['system'],
        IntentType.CUSTOMER_ANALYSIS: ['risk'],
        IntentType.GENERAL: []
    }
    active_domain = domain_mapping.get(intent, [])
    
    return IntentResult(
        intent=intent,
        confidence=confidence,
        reasoning=f"降级处理：基于关键词匹配（政策:{policy_score}, 系统:{system_score}, 客户:{customer_score}）",
        route_to=route_to,
        active_domain=active_domain,  # 使用规则映射
        business_object='',  # 降级处理时无法抽取业务对象
        operation_stage=''   # 降级处理时无法抽取操作阶段
    )

# ============================================================================
# 路由决策
# ============================================================================

def route_query(intent_result: IntentResult, question: str) -> Dict:
    """
    根据意图识别结果进行路由决策
    
    参数:
        intent_result: 意图识别结果
        question: 用户问题
    
    返回:
        Dict: 路由信息
    """
    route_info = {
        'intent': intent_result.intent.value,
        'route_to': intent_result.route_to,
        'confidence': intent_result.confidence,
        'reasoning': intent_result.reasoning,
        'question': question
    }
    
    # 根据路由目标设置参数
    if intent_result.route_to == 'rag_policy':
        route_info['domain'] = 'policy'
        route_info['module'] = 'rag'
    elif intent_result.route_to == 'rag_system':
        route_info['domain'] = 'system'
        route_info['module'] = 'rag'
    elif intent_result.route_to == 'prediction':
        route_info['module'] = 'prediction'
    else:
        route_info['module'] = 'general'
    
    return route_info

# ============================================================================
# 主函数（用于测试）
# ============================================================================

def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='问题理解与意图识别测试')
    parser.add_argument('question', type=str, help='用户问题')
    parser.add_argument('--no-cot', action='store_true', help='禁用CoT思维链推理')
    
    args = parser.parse_args()
    
    try:
        print(f"\n{'='*60}")
        print(f"问题理解与意图识别测试")
        print(f"{'='*60}")
        print(f"用户问题: {args.question}")
        print(f"使用CoT: {not args.no_cot}\n")
        
        # 进行问题理解与意图识别
        result = classify_intent(args.question, use_cot=not args.no_cot)
        
        # 路由决策
        route_info = route_query(result, args.question)
        
        # 显示结果
        print(f"\n{'='*60}")
        print(f"识别结果")
        print(f"{'='*60}")
        print(f"意图类型: {result.intent.value}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"路由目标: {result.route_to}")
        print(f"处理模块: {route_info['module']}")
        if 'domain' in route_info:
            print(f"域类型: {route_info['domain']}")
        
        print(f"\n推理过程（CoT）:")
        print(f"{'-'*60}")
        print(result.reasoning)
        
    except Exception as e:
        print(f"\n❌ 问题理解与意图识别失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

