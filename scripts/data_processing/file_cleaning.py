"""
文件清洗脚本
功能：
1. 识别需要OCR处理的图片文件（PNG、JPG等）
2. 使用OCR提取文本并恢复结构（保证条款级连续性和可追溯性）
3. 将OCR结果合并成一个文档
4. 将不需要清洗的文件直接复制到processed目录
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import re
import warnings

# 设置环境变量，跳过模型源检查（加快初始化速度）
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# OCR相关库（需要安装：pip install paddlepaddle paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple）
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("警告: PaddleOCR未安装，将使用备用方案")

# 图片文件扩展名
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
# 文档文件扩展名（直接复制）
DOC_EXTENSIONS = {'.docx', '.doc', '.pdf', '.txt', '.md'}


def init_ocr():
    """
    初始化OCR引擎
    使用PaddleOCR，支持中英文识别
    """
    if not PADDLEOCR_AVAILABLE:
        raise ImportError("请先安装PaddleOCR: pip install paddlepaddle paddleocr")
    
    print("正在初始化OCR引擎（首次运行需要下载模型，请稍候...）")
    
    # 初始化OCR，使用中文模型
    # use_angle_cls=True 表示使用文字方向分类器
    # lang='ch' 表示使用中文模型
    # use_gpu=False 表示使用CPU（如果GPU不可用）
    # show_log=False 不显示详细日志
    try:
        # 初始化OCR，使用中文模型
        # use_angle_cls=True 表示使用文字方向分类器（初始化时设置）
        # lang='ch' 表示使用中文模型
        print("  提示: 首次运行会下载模型文件，请耐心等待...")
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("✓ OCR引擎初始化成功")
        return ocr
    except Exception as e:
        print(f"OCR初始化出错: {e}")
        print("提示: 如果网络连接有问题，请检查网络或稍后重试")
        raise


def extract_text_from_image(ocr, image_path: str) -> List[Tuple[str, float]]:
    """
    从单张图片中提取文本
    兼容新旧版本的PaddleOCR API（支持OCRResult对象格式）
    
    参数:
        ocr: OCR引擎实例
        image_path: 图片路径
    
    返回:
        List[Tuple[str, float]]: [(文本内容, 置信度), ...]
    """
    try:
        # 抑制DeprecationWarning（PaddleOCR建议使用predict，但ocr方法仍然可用）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            # 调用OCR识别
            result = ocr.ocr(image_path)
        
        if result is None or len(result) == 0:
            return []
        
        # result[0] 可能是OCRResult对象（新版本）或列表（旧版本）
        ocr_result = result[0]
        texts = []
        
        # 方法1: 处理OCRResult对象格式（新版本PaddleOCR 3.x）
        # OCRResult是字典类型，包含 rec_texts 和 rec_scores 键
        if hasattr(ocr_result, 'keys') and 'rec_texts' in ocr_result:
            try:
                rec_texts = ocr_result['rec_texts']
                rec_scores = ocr_result.get('rec_scores', [])
                
                # 处理scores（可能是numpy数组）
                try:
                    import numpy as np
                    if isinstance(rec_scores, np.ndarray):
                        rec_scores = rec_scores.tolist()
                except ImportError:
                    pass
                
                # 确保scores长度与texts匹配
                if not isinstance(rec_scores, list):
                    rec_scores = []
                if len(rec_scores) != len(rec_texts):
                    rec_scores = [1.0] * len(rec_texts)
                
                # 提取文本和置信度
                for text, score in zip(rec_texts, rec_scores):
                    if text and str(text).strip():
                        confidence = float(score) if score is not None else 1.0
                        texts.append((str(text).strip(), confidence))
                
                return texts
            except Exception as e:
                # 如果直接访问失败，尝试其他方法
                pass
        
        # 方法2: 从json属性中提取（OCRResult对象）
        if hasattr(ocr_result, 'json'):
            try:
                json_data = ocr_result.json
                if isinstance(json_data, dict) and 'res' in json_data:
                    res_dict = json_data['res']
                    if isinstance(res_dict, dict) and 'rec_texts' in res_dict:
                        rec_texts = res_dict['rec_texts']
                        rec_scores = res_dict.get('rec_scores', [])
                        
                        # 处理scores
                        try:
                            import numpy as np
                            if isinstance(rec_scores, np.ndarray):
                                rec_scores = rec_scores.tolist()
                        except ImportError:
                            pass
                        
                        if not isinstance(rec_scores, list):
                            rec_scores = []
                        if len(rec_scores) != len(rec_texts):
                            rec_scores = [1.0] * len(rec_texts)
                        
                        for text, score in zip(rec_texts, rec_scores):
                            if text and str(text).strip():
                                confidence = float(score) if score is not None else 1.0
                                texts.append((str(text).strip(), confidence))
                        
                        if texts:
                            return texts
            except Exception:
                pass
        
        # 方法3: 处理标准列表格式（旧版本PaddleOCR）
        # 格式：[[[坐标], (文本, 置信度)], ...]
        if isinstance(ocr_result, list):
            for line in ocr_result:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                        text = str(text_info[0]).strip()
                        confidence = float(text_info[1]) if len(text_info) > 1 else 1.0
                        if text:
                            texts.append((text, confidence))
            return texts
        
        return texts
        
    except Exception as e:
        print(f"      ⚠ OCR处理出错: {os.path.basename(image_path)} - {e}")
        return []


def merge_texts_with_structure(texts_list: List[List[Tuple[str, float]]], 
                                image_names: List[str],
                                folder_name: str = "") -> str:
    """
    合并多张图片的文本，保持结构（条款级连续性）
    
    参数:
        texts_list: 每张图片的文本列表
        image_names: 对应的图片文件名（用于追溯）
        folder_name: 文件夹名称（用于文档标题）
    
    返回:
        str: 合并后的文本内容
    """
    merged_content = []
    
    # 添加文档头部信息
    if folder_name:
        merged_content.append(f"# {folder_name}\n")
    else:
        merged_content.append("# OCR恢复文档\n")
    merged_content.append("> 本文档由多张图片OCR识别后合并生成\n\n")
    merged_content.append("---\n\n")
    
    # 遍历每张图片的文本
    for idx, (texts, img_name) in enumerate(zip(texts_list, image_names), 1):
        # 提取文本内容
        page_texts = []
        for text, confidence in texts:
            # 过滤低置信度的文本（可选）
            if confidence > 0.5:  # 只保留置信度大于0.5的文本
                page_texts.append(text)
        
        # 合并当前页的文本
        page_content = " ".join(page_texts)
        
        # 简单的段落处理：如果文本以数字开头，可能是条款，添加换行
        # 例如："第一条"、"1."、"（一）"等
        page_content = re.sub(r'([。！？])\s*([第\d一二三四五六七八九十]+[条款项])', r'\1\n\n\2', page_content)
        page_content = re.sub(r'([。！？])\s*([（(][一二三四五六七八九十]+[）)])', r'\1\n\n\2', page_content)
        page_content = re.sub(r'([。！？])\s*(\d+[\.、])', r'\1\n\n\2', page_content)
        
        merged_content.append(page_content)
        
        # 如果不是最后一张图片，添加分隔
        if idx < len(texts_list):
            merged_content.append("\n\n")
    
    return "".join(merged_content)


def extract_page_number_from_text(texts: List[Tuple[str, float]]) -> int:
    """
    从OCR识别的文本中提取页码
    
    参数:
        texts: OCR识别的文本列表 [(文本, 置信度), ...]
    
    返回:
        int: 页码，如果未找到则返回-1
    """
    # 合并所有文本
    full_text = " ".join([text for text, _ in texts])
    
    # 匹配页码的多种格式：
    # 1. "-1-", "-2-", "-10-" 等（最常见的格式）
    # 2. "第1页", "第2页" 等
    # 3. "1/16", "2/16" 等
    # 4. 单独的页码数字（在文本末尾或开头）
    
    # 优先匹配 "-数字-" 格式（最常见）
    page_match = re.search(r'-(\d+)-', full_text)
    if page_match:
        return int(page_match.group(1))
    
    # 匹配 "第数字页" 格式
    page_match = re.search(r'第(\d+)页', full_text)
    if page_match:
        return int(page_match.group(1))
    
    # 匹配 "数字/数字" 格式（如 "1/16"）
    page_match = re.search(r'(\d+)/(\d+)', full_text)
    if page_match:
        return int(page_match.group(1))
    
    # 匹配文本末尾的单独数字（可能是页码）
    # 检查文本最后几行是否有单独的数字
    lines = full_text.split()
    if lines:
        # 检查最后几个词
        for word in reversed(lines[-5:]):
            # 如果是纯数字且小于1000（页码通常不会太大）
            if word.isdigit() and int(word) < 1000:
                return int(word)
    
    return -1  # 未找到页码


def sort_images_by_page_number(image_files: List[str], ocr) -> tuple:
    """
    对图片文件进行排序（按OCR识别的页码顺序）
    先对每张图片进行OCR识别，提取页码，然后按页码排序
    同时保存OCR结果，避免重复识别
    
    参数:
        image_files: 图片文件路径列表
        ocr: OCR引擎实例
    
    返回:
        tuple: (排序后的文件路径列表, OCR结果字典 {文件路径: texts})
    """
    print(f"  正在识别页码以确定排序顺序...")
    
    # 对每张图片进行OCR识别，提取页码，同时保存OCR结果
    image_page_pairs = []
    ocr_results_cache = {}  # 缓存OCR结果，避免重复识别
    
    # 去重处理（避免同一文件被处理多次）
    processed_files = set()
    
    for img_path in image_files:
        # 标准化路径（解决Windows路径大小写问题）
        img_path_normalized = os.path.normpath(img_path)
        
        # 如果已经处理过，跳过
        if img_path_normalized in processed_files:
            continue
        
        processed_files.add(img_path_normalized)
        img_name = os.path.basename(img_path)
        
        # OCR识别（同时用于提取页码和后续使用）
        texts = extract_text_from_image(ocr, img_path)
        ocr_results_cache[img_path_normalized] = texts  # 缓存结果（使用标准化路径）
        
        page_num = extract_page_number_from_text(texts)
        
        if page_num == -1:
            # 如果未找到页码，使用文件名中的数字作为备用
            numbers = re.findall(r'\d+', img_name)
            if numbers:
                last_num = int(numbers[-1])
                if last_num < 1000:
                    page_num = last_num
                elif len(numbers) >= 2:
                    page_num = int(numbers[-2])
                else:
                    page_num = 9999
            else:
                page_num = 9999  # 未找到页码的放在最后
        
        image_page_pairs.append((page_num, img_path_normalized))
        print(f"    {img_name}: 页码 {page_num}")
    
    # 按页码排序
    image_page_pairs.sort(key=lambda x: x[0])
    
    # 返回排序后的文件路径列表和OCR结果缓存
    sorted_files = [img_path for _, img_path in image_page_pairs]
    return sorted_files, ocr_results_cache


def process_image_folder(folder_path: Path, output_path: Path, ocr) -> bool:
    """
    处理包含多张图片的文件夹
    按图片文件名中的数字顺序处理，合并成一个文档
    
    参数:
        folder_path: 图片文件夹路径
        output_path: 输出文件路径（.md文件）
        ocr: OCR引擎
    
    返回:
        bool: 是否成功
    """
    # 获取所有图片文件（使用set去重，避免重复处理）
    image_files_set = set()
    for ext in IMAGE_EXTENSIONS:
        image_files_set.update(folder_path.glob(f'*{ext}'))
        image_files_set.update(folder_path.glob(f'*{ext.upper()}'))
    
    if not image_files_set:
        print(f"  文件夹中没有找到图片文件")
        return False
    
    # 转换为字符串路径并去重
    image_files = list(set(str(f) for f in image_files_set))
    
    print(f"  找到 {len(image_files)} 张图片")
    
    # 按页码排序（先OCR识别提取页码，同时缓存OCR结果）
    image_files, ocr_cache = sort_images_by_page_number(image_files, ocr)
    
    print(f"\n  按页码排序完成，使用缓存的OCR结果...")
    
    # 使用缓存的OCR结果（避免重复识别）
    all_texts = []
    image_names = []
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        print(f"    [{idx}/{len(image_files)}] 处理: {img_name}", end="", flush=True)
        
        # 标准化路径（与排序时保持一致）
        img_path_normalized = os.path.normpath(img_path)
        
        # 使用缓存的OCR结果
        texts = ocr_cache.get(img_path_normalized, [])
        
        if texts:
            print(f" ✓ (识别到 {len(texts)} 行文本)")
        else:
            print(f" ⚠ (未识别到文本)")
        
        all_texts.append(texts)
        image_names.append(img_name)
    
    # 检查是否有文本被提取
    total_texts = sum(len(texts) for texts in all_texts)
    if total_texts == 0:
        print(f"  ⚠ 警告: 所有图片都未识别到文本，请检查图片质量或OCR配置")
    
    # 合并文本（使用文件夹名称作为标题）
    folder_name = folder_path.name
    print(f"  正在合并文本并恢复结构... (共提取 {total_texts} 行文本)")
    merged_text = merge_texts_with_structure(all_texts, image_names, folder_name)
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_text)
    
    print(f"  ✓ OCR处理完成，已保存到: {output_path.name}")
    return True


def copy_document_file(src_path: Path, dst_path: Path) -> bool:
    """
    复制文档文件到processed目录
    
    参数:
        src_path: 源文件路径
        dst_path: 目标文件路径
    
    返回:
        bool: 是否成功
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"✓ 已复制: {src_path.name} -> {dst_path}")
        return True
    except Exception as e:
        print(f"✗ 复制文件 {src_path.name} 失败: {e}")
        return False


def clean_folder_images(raw_dir: Path, processed_dir: Path):
    """
    清洗文件夹内的图片（OCR处理）
    只处理文件夹内的图片，按数字顺序合并，用文件夹名称命名
    
    参数:
        raw_dir: 原始文件目录 (data/raw/policy 或 data/raw/system)
        processed_dir: 处理后文件目录 (data/processed/policy 或 data/processed/system)
    """
    print("=" * 60)
    print(f"开始OCR处理 - {raw_dir.name.upper()} 目录")
    print("=" * 60)
    
    # 检查是否有包含图片的文件夹
    folders_with_images = []
    for item in raw_dir.iterdir():
        if item.is_dir():
            # 检查文件夹中是否有图片
            has_images = False
            for ext in IMAGE_EXTENSIONS:
                if list(item.glob(f'*{ext}')):
                    has_images = True
                    break
            if has_images:
                folders_with_images.append(item)
    
    if not folders_with_images:
        print(f"未找到包含图片的文件夹，跳过OCR处理")
        return
    
    # 初始化OCR引擎
    print(f"\n检测到 {len(folders_with_images)} 个包含图片的文件夹，初始化OCR引擎...")
    try:
        ocr = init_ocr()
        print("✓ OCR引擎初始化成功\n")
    except Exception as e:
        print(f"✗ OCR引擎初始化失败: {e}")
        print("  请安装: pip install paddlepaddle paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return
    
    # 处理每个包含图片的文件夹
    for folder_path in folders_with_images:
        folder_name = folder_path.name
        print(f"\n处理文件夹: {folder_name}")
        
        # 使用文件夹名称作为输出文件名（.md格式）
        output_filename = f"{folder_name}.md"
        output_path = processed_dir / output_filename
        
        # 处理图片文件夹
        process_image_folder(folder_path, output_path, ocr)
    
    print("\n" + "=" * 60)
    print(f"OCR处理完成！")
    print("=" * 60)


def main():
    """
    主函数
    处理 policy 和 system 两个目录下的文件夹图片OCR
    """
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    
    # 处理 policy 和 system 两个目录
    domains = ['policy', 'system']
    
    for domain in domains:
        raw_dir = project_root / "data" / "raw" / domain
        processed_dir = project_root / "data" / "processed" / domain
        
        # 检查原始目录是否存在
        if not raw_dir.exists():
            print(f"⚠ 跳过: 原始目录不存在: {raw_dir}")
            continue
        
        # 执行OCR处理
        clean_folder_images(raw_dir, processed_dir)
        
        if domain != domains[-1]:
            print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

