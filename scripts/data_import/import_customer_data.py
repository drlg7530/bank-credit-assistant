"""
客户数据导入脚本
功能：读取CSV文件并将客户数据导入到MySQL数据库

使用说明：
1. 确保MySQL数据库已创建并执行了建表语句
2. 确保CSV文件路径正确
3. 运行脚本：python scripts/data_import/import_customer_data.py
"""

import sys
import os
import csv
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到Python路径，以便导入config模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pymysql
    from pymysql import Error
except ImportError:
    print("错误：未安装pymysql库，请先安装：pip install pymysql")
    sys.exit(1)

# 导入数据库配置
from config.database import DB_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_import.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clean_value(value: str) -> Optional[Any]:
    """
    清洗CSV数据值
    将空字符串转换为None，处理数字类型
    
    参数:
        value: CSV中的原始值（字符串格式）
    
    返回:
        清洗后的值，空值返回None
    """
    if value is None or value.strip() == '':
        return None
    return value.strip()


def parse_dependents(value: str) -> Optional[str]:
    """
    解析家属人数字段
    处理"3+"这样的特殊值
    
    参数:
        value: 家属人数原始值
    
    返回:
        处理后的家属人数字符串
    """
    if value is None or value.strip() == '':
        return None
    return value.strip()


def parse_numeric(value: str, default: Optional[float] = None) -> Optional[float]:
    """
    解析数字字段
    将字符串转换为浮点数，空值返回None
    
    参数:
        value: 数字字符串
        default: 默认值（如果解析失败）
    
    返回:
        解析后的数字或None
    """
    if value is None or value.strip() == '':
        return default
    try:
        return float(value.strip())
    except ValueError:
        logger.warning(f"无法解析数字值: {value}，使用默认值: {default}")
        return default


def parse_integer(value: str, default: Optional[int] = None) -> Optional[int]:
    """
    解析整数字段
    将字符串转换为整数，空值返回None
    
    参数:
        value: 整数字符串
        default: 默认值（如果解析失败）
    
    返回:
        解析后的整数或None
    """
    if value is None or value.strip() == '':
        return default
    try:
        return int(float(value.strip()))  # 先转float再转int，处理"3.0"这种情况
    except ValueError:
        logger.warning(f"无法解析整数值: {value}，使用默认值: {default}")
        return default


def connect_database() -> Optional[pymysql.Connection]:
    """
    连接MySQL数据库
    
    返回:
        数据库连接对象，失败返回None
    """
    try:
        connection = pymysql.connect(**DB_CONFIG)
        logger.info(f"成功连接到数据库: {DB_CONFIG['database']}")
        return connection
    except Error as e:
        logger.error(f"数据库连接失败: {e}")
        return None


def read_csv_data(csv_path: str) -> list:
    """
    读取CSV文件数据
    
    参数:
        csv_path: CSV文件路径
    
    返回:
        数据行列表，每行是一个字典
    """
    data_rows = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # 使用csv.DictReader自动将第一行作为字段名
            reader = csv.DictReader(csvfile)
            
            for row_num, row in enumerate(reader, start=2):  # 从第2行开始计数（第1行是表头）
                try:
                    # 清洗和转换数据
                    processed_row = {
                        'Loan_ID': clean_value(row.get('Loan_ID')),
                        'Gender': clean_value(row.get('Gender')),
                        'Married': clean_value(row.get('Married')),
                        'Dependents': parse_dependents(row.get('Dependents')),
                        'Education': clean_value(row.get('Education')),
                        'Self_Employed': clean_value(row.get('Self_Employed')),
                        'ApplicantIncome': parse_numeric(row.get('ApplicantIncome'), 0),
                        'CoapplicantIncome': parse_numeric(row.get('CoapplicantIncome'), 0),
                        'LoanAmount': parse_numeric(row.get('LoanAmount')),  # 可能为空
                        'Loan_Amount_Term': parse_integer(row.get('Loan_Amount_Term')),
                        'Credit_History': parse_integer(row.get('Credit_History')),
                        'Property_Area': clean_value(row.get('Property_Area')),
                        'Loan_Status': clean_value(row.get('Loan_Status')),
                    }
                    
                    # 验证必填字段
                    if not processed_row['Loan_ID']:
                        logger.warning(f"第{row_num}行：Loan_ID为空，跳过该行")
                        continue
                    
                    data_rows.append(processed_row)
                    
                except Exception as e:
                    logger.error(f"第{row_num}行数据处理失败: {e}")
                    continue
            
            logger.info(f"成功读取 {len(data_rows)} 条数据记录")
            return data_rows
            
    except FileNotFoundError:
        logger.error(f"文件不存在: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        return []


def insert_data_batch(connection: pymysql.Connection, data_rows: list, batch_size: int = 100) -> int:
    """
    批量插入数据到数据库
    
    参数:
        connection: 数据库连接对象
        data_rows: 要插入的数据行列表
        batch_size: 每批插入的记录数
    
    返回:
        成功插入的记录数
    """
    if not data_rows:
        logger.warning("没有数据需要插入")
        return 0
    
    # SQL插入语句
    insert_sql = """
    INSERT INTO customer_loan_data (
        Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
        Credit_History, Property_Area, Loan_Status
    ) VALUES (
        %(Loan_ID)s, %(Gender)s, %(Married)s, %(Dependents)s, %(Education)s, %(Self_Employed)s,
        %(ApplicantIncome)s, %(CoapplicantIncome)s, %(LoanAmount)s, %(Loan_Amount_Term)s,
        %(Credit_History)s, %(Property_Area)s, %(Loan_Status)s
    )
    ON DUPLICATE KEY UPDATE
        Gender = VALUES(Gender),
        Married = VALUES(Married),
        Dependents = VALUES(Dependents),
        Education = VALUES(Education),
        Self_Employed = VALUES(Self_Employed),
        ApplicantIncome = VALUES(ApplicantIncome),
        CoapplicantIncome = VALUES(CoapplicantIncome),
        LoanAmount = VALUES(LoanAmount),
        Loan_Amount_Term = VALUES(Loan_Amount_Term),
        Credit_History = VALUES(Credit_History),
        Property_Area = VALUES(Property_Area),
        Loan_Status = VALUES(Loan_Status)
    """
    
    success_count = 0
    error_count = 0
    
    try:
        cursor = connection.cursor()
        
        # 分批插入数据
        for i in range(0, len(data_rows), batch_size):
            batch = data_rows[i:i + batch_size]
            
            try:
                # 执行批量插入
                cursor.executemany(insert_sql, batch)
                connection.commit()
                
                success_count += len(batch)
                logger.info(f"成功插入 {len(batch)} 条记录（总计: {success_count}/{len(data_rows)}）")
                
            except Error as e:
                connection.rollback()
                error_count += len(batch)
                logger.error(f"批量插入失败（第{i//batch_size + 1}批）: {e}")
                # 尝试逐条插入，找出问题记录
                for record in batch:
                    try:
                        cursor.execute(insert_sql, record)
                        connection.commit()
                        success_count += 1
                        error_count -= 1
                    except Error as err:
                        logger.error(f"插入记录失败 (Loan_ID: {record.get('Loan_ID')}): {err}")
                        connection.rollback()
        
        cursor.close()
        logger.info(f"数据导入完成：成功 {success_count} 条，失败 {error_count} 条")
        return success_count
        
    except Error as e:
        logger.error(f"数据库操作失败: {e}")
        connection.rollback()
        return success_count


def main():
    """
    主函数：执行数据导入流程
    """
    logger.info("=" * 60)
    logger.info("开始执行客户数据导入")
    logger.info("=" * 60)
    
    # 步骤1：确定CSV文件路径
    csv_file_path = project_root / 'data' / 'customer' / 'train.csv'
    
    if not csv_file_path.exists():
        logger.error(f"CSV文件不存在: {csv_file_path}")
        return
    
    logger.info(f"CSV文件路径: {csv_file_path}")
    
    # 步骤2：读取CSV数据
    logger.info("步骤1：读取CSV文件数据...")
    data_rows = read_csv_data(str(csv_file_path))
    
    if not data_rows:
        logger.error("没有读取到有效数据，导入终止")
        return
    
    # 步骤3：连接数据库
    logger.info("步骤2：连接数据库...")
    connection = connect_database()
    
    if not connection:
        logger.error("数据库连接失败，导入终止")
        return
    
    try:
        # 步骤4：插入数据
        logger.info("步骤3：开始插入数据到数据库...")
        success_count = insert_data_batch(connection, data_rows)
        
        if success_count > 0:
            logger.info(f"数据导入成功！共导入 {success_count} 条记录")
        else:
            logger.warning("没有成功导入任何数据")
            
    finally:
        # 步骤5：关闭数据库连接
        connection.close()
        logger.info("数据库连接已关闭")
    
    logger.info("=" * 60)
    logger.info("数据导入流程完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

