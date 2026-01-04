"""
客户数据查询模块
功能：从MySQL数据库查询客户信息，用于LightGBM模型预测

使用说明：
1. 确保MySQL数据库已连接并包含客户数据
2. 运行查询：python src/prediction/data_query.py
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

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
        logging.FileHandler('logs/data_query.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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


def query_all_customers(connection: pymysql.Connection, limit: Optional[int] = None) -> pd.DataFrame:
    """
    查询所有客户数据
    
    参数:
        connection: 数据库连接对象
        limit: 限制返回的记录数，None表示返回所有记录
    
    返回:
        包含客户数据的DataFrame
    """
    try:
        # 构建SQL查询语句
        if limit:
            sql = f"""
            SELECT 
                Loan_ID,
                Gender,
                Married,
                Dependents,
                Education,
                Self_Employed,
                ApplicantIncome,
                CoapplicantIncome,
                LoanAmount,
                Loan_Amount_Term,
                Credit_History,
                Property_Area,
                Loan_Status,
                created_at,
                updated_at
            FROM customer_loan_data
            LIMIT {limit}
            """
        else:
            sql = """
            SELECT 
                Loan_ID,
                Gender,
                Married,
                Dependents,
                Education,
                Self_Employed,
                ApplicantIncome,
                CoapplicantIncome,
                LoanAmount,
                Loan_Amount_Term,
                Credit_History,
                Property_Area,
                Loan_Status,
                created_at,
                updated_at
            FROM customer_loan_data
            """
        
        # 使用pandas读取SQL查询结果
        df = pd.read_sql(sql, connection)
        logger.info(f"成功查询到 {len(df)} 条客户记录")
        return df
        
    except Error as e:
        logger.error(f"查询客户数据失败: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"查询过程发生错误: {e}")
        return pd.DataFrame()


def query_customers_by_condition(
    connection: pymysql.Connection,
    conditions: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    根据条件查询客户数据
    
    参数:
        connection: 数据库连接对象
        conditions: 查询条件字典，例如 {'Gender': 'Male', 'Loan_Status': 'Y'}
        limit: 限制返回的记录数
    
    返回:
        包含客户数据的DataFrame
    """
    try:
        # 构建基础SQL查询语句
        sql = """
        SELECT 
            Loan_ID,
            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area,
            Loan_Status,
            created_at,
            updated_at
        FROM customer_loan_data
        WHERE 1=1
        """
        
        # 添加查询条件
        params = []
        if conditions:
            for key, value in conditions.items():
                if value is not None:
                    sql += f" AND {key} = %s"
                    params.append(value)
        
        # 添加限制条件
        if limit:
            sql += f" LIMIT {limit}"
        
        # 执行查询
        df = pd.read_sql(sql, connection, params=params if params else None)
        logger.info(f"根据条件查询到 {len(df)} 条客户记录")
        return df
        
    except Error as e:
        logger.error(f"条件查询客户数据失败: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"查询过程发生错误: {e}")
        return pd.DataFrame()


def query_customer_statistics(connection: pymysql.Connection) -> Dict[str, Any]:
    """
    查询客户数据统计信息
    
    参数:
        connection: 数据库连接对象
    
    返回:
        包含统计信息的字典
    """
    try:
        # 查询总记录数
        sql_count = "SELECT COUNT(*) as total FROM customer_loan_data"
        df_count = pd.read_sql(sql_count, connection)
        total_count = df_count['total'].iloc[0]
        
        # 查询各字段的统计信息
        sql_stats = """
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT Gender) as gender_types,
            COUNT(DISTINCT Married) as married_types,
            COUNT(DISTINCT Education) as education_types,
            COUNT(DISTINCT Property_Area) as area_types,
            AVG(ApplicantIncome) as avg_applicant_income,
            AVG(CoapplicantIncome) as avg_coapplicant_income,
            AVG(LoanAmount) as avg_loan_amount,
            AVG(Loan_Amount_Term) as avg_loan_term,
            SUM(CASE WHEN Loan_Status = 'Y' THEN 1 ELSE 0 END) as approved_count,
            SUM(CASE WHEN Loan_Status = 'N' THEN 1 ELSE 0 END) as rejected_count
        FROM customer_loan_data
        """
        
        df_stats = pd.read_sql(sql_stats, connection)
        stats = df_stats.iloc[0].to_dict()
        
        logger.info(f"数据统计信息查询成功，总记录数: {total_count}")
        return stats
        
    except Error as e:
        logger.error(f"查询统计信息失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"统计查询过程发生错误: {e}")
        return {}


def save_to_csv(df: pd.DataFrame, output_path: str) -> bool:
    """
    将DataFrame保存为CSV文件
    
    参数:
        df: 要保存的DataFrame
        output_path: 输出文件路径
    
    返回:
        保存是否成功
    """
    try:
        # 确保输出目录存在
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存到: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")
        return False


def main():
    """
    主函数：演示数据查询功能
    """
    logger.info("=" * 60)
    logger.info("开始查询客户数据")
    logger.info("=" * 60)
    
    # 步骤1：连接数据库
    logger.info("步骤1：连接数据库...")
    connection = connect_database()
    
    if not connection:
        logger.error("数据库连接失败，查询终止")
        return
    
    try:
        # 步骤2：查询数据统计信息
        logger.info("步骤2：查询数据统计信息...")
        stats = query_customer_statistics(connection)
        if stats:
            logger.info(f"数据统计: {stats}")
        
        # 步骤3：查询所有客户数据（限制前1000条）
        logger.info("步骤3：查询客户数据（前1000条）...")
        df = query_all_customers(connection, limit=1000)
        
        if not df.empty:
            logger.info(f"查询成功，共 {len(df)} 条记录")
            logger.info(f"数据列: {list(df.columns)}")
            logger.info(f"\n数据预览（前5条）:\n{df.head()}")
            
            # 步骤4：保存查询结果到CSV（可选）
            output_path = project_root / 'data' / 'customer' / 'query_result.csv'
            if save_to_csv(df, str(output_path)):
                logger.info(f"查询结果已保存到: {output_path}")
        else:
            logger.warning("未查询到任何数据")
        
        # 步骤5：示例：根据条件查询
        logger.info("\n步骤4：示例条件查询（性别为Male的客户）...")
        df_filtered = query_customers_by_condition(
            connection,
            conditions={'Gender': 'Male'},
            limit=10
        )
        if not df_filtered.empty:
            logger.info(f"条件查询成功，共 {len(df_filtered)} 条记录")
            logger.info(f"\n条件查询结果预览:\n{df_filtered.head()}")
        
    finally:
        # 步骤6：关闭数据库连接
        connection.close()
        logger.info("数据库连接已关闭")
    
    logger.info("=" * 60)
    logger.info("数据查询流程完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

