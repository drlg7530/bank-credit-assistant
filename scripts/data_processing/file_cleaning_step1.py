"""
文件清洗脚本 - 第一阶段：复制第一层文件
功能：将raw目录下第一层的所有文件直接复制到processed目录（不限制文件类型）
说明：文件夹将在第二阶段使用OCR处理（如果包含图片）
"""

import os
import shutil
from pathlib import Path


def copy_file(src_path: Path, dst_path: Path) -> bool:
    """
    复制文件到processed目录
    
    参数:
        src_path: 源文件路径
        dst_path: 目标文件路径
    
    返回:
        bool: 是否成功
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"✓ 已复制: {src_path.name}")
        return True
    except Exception as e:
        print(f"✗ 复制文件 {src_path.name} 失败: {e}")
        return False


def process_first_level_files(raw_dir: Path, processed_dir: Path):
    """
    处理第一层文件（不限制文件类型，所有文件都复制）
    跳过文件夹，文件夹将在第二阶段处理
    
    参数:
        raw_dir: 原始文件目录 (data/raw/policy 或 data/raw/system)
        processed_dir: 处理后文件目录 (data/processed/policy 或 data/processed/system)
    """
    print("=" * 60)
    print(f"文件清洗 - 第一阶段：复制第一层文件")
    print(f"源目录: {raw_dir.name}")
    print("=" * 60)
    print("说明：所有第一层文件直接复制，文件夹将在第二阶段处理\n")
    
    # 遍历原始目录中的所有文件和文件夹
    for item in raw_dir.iterdir():
        if item.is_file():
            # 第一层的所有文件都直接复制（不限制文件类型）
            dst_path = processed_dir / item.name
            copy_file(item, dst_path)
        elif item.is_dir():
            # 跳过文件夹，等待第二阶段OCR处理
            print(f"⏭ 跳过文件夹（待OCR处理）: {item.name}")
    
    print("\n" + "=" * 60)
    print(f"第一阶段完成！第一层文件已复制到 processed/{raw_dir.name} 目录")
    print("=" * 60)
    print("\n提示：文件夹需要使用 file_cleaning.py 进行OCR处理")


def main():
    """
    主函数
    处理 policy 和 system 两个目录
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
        
        print(f"\n{'='*60}")
        print(f"处理 {domain.upper()} 目录")
        print(f"{'='*60}\n")
        
        # 执行第一阶段处理
        process_first_level_files(raw_dir, processed_dir)


if __name__ == "__main__":
    main()

