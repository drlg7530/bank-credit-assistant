"""
特征提取模块
功能：从客户数据中提取LightGBM模型预测所需的特征

特征工程说明：
1. 原始特征：直接使用数据库中的字段
2. 衍生特征：基于原始特征计算的新特征（如收入比、贷款收入比等）
3. 编码特征：将分类特征转换为数值特征（Label Encoding / One-Hot Encoding）

使用说明：
1. 确保已安装pandas、numpy、sklearn
2. 运行特征提取：python src/prediction/feature_extraction.py
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入数据查询模块
from src.prediction.data_query import connect_database, query_all_customers

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理缺失值
    
    参数:
        df: 原始数据DataFrame
    
    返回:
        处理缺失值后的DataFrame
    """
    df_processed = df.copy()
    
    # 数值型字段：用中位数填充
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                   'Loan_Amount_Term', 'Credit_History']
    for col in numeric_cols:
        if col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                median_value = df_processed[col].median()
                df_processed[col].fillna(median_value, inplace=True)
                logger.info(f"字段 {col} 缺失值已用中位数 {median_value} 填充")
    
    # 分类型字段：用众数填充
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                        'Self_Employed', 'Property_Area']
    for col in categorical_cols:
        if col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col].fillna(mode_value, inplace=True)
                logger.info(f"字段 {col} 缺失值已用众数 {mode_value} 填充")
    
    return df_processed


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建衍生特征
    
    参数:
        df: 原始数据DataFrame
    
    返回:
        包含衍生特征的DataFrame
    """
    df_features = df.copy()
    
    # 检查LoanAmount是否存在（预测时可能不存在）
    has_loan_amount = 'LoanAmount' in df_features.columns
    
    # 特征1：总家庭收入 = 申请人收入 + 共同申请人收入
    df_features['TotalIncome'] = (
        df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
    )
    logger.info("已创建特征: TotalIncome (总家庭收入)")
    
    # 特征2：贷款收入比 = 贷款金额 / 总家庭收入（如果总收入为0，设为0）
    # 注意：预测时LoanAmount不存在，设为0
    if has_loan_amount:
        df_features['LoanToIncomeRatio'] = np.where(
            df_features['TotalIncome'] > 0,
            df_features['LoanAmount'] / df_features['TotalIncome'],
            0
        )
    else:
        # 预测时LoanAmount不存在，设为0
        df_features['LoanToIncomeRatio'] = 0
    logger.info("已创建特征: LoanToIncomeRatio (贷款收入比)")
    
    # 特征3：月收入 = 总家庭收入 / 12
    df_features['MonthlyIncome'] = df_features['TotalIncome'] / 12
    logger.info("已创建特征: MonthlyIncome (月收入)")
    
    # 特征4：月还款额估算 = 贷款金额 / 贷款期限（天转月，假设30天/月）
    # 注意：预测时LoanAmount不存在，设为0
    if has_loan_amount:
        df_features['MonthlyPayment'] = np.where(
            df_features['Loan_Amount_Term'] > 0,
            df_features['LoanAmount'] / (df_features['Loan_Amount_Term'] / 30),
            0
        )
    else:
        # 预测时LoanAmount不存在，设为0
        df_features['MonthlyPayment'] = 0
    logger.info("已创建特征: MonthlyPayment (月还款额估算)")
    
    # 特征5：还款收入比 = 月还款额 / 月收入
    df_features['PaymentToIncomeRatio'] = np.where(
        df_features['MonthlyIncome'] > 0,
        df_features['MonthlyPayment'] / df_features['MonthlyIncome'],
        0
    )
    logger.info("已创建特征: PaymentToIncomeRatio (还款收入比)")
    
    # 特征6：是否有共同申请人（二值特征）
    df_features['HasCoapplicant'] = (df_features['CoapplicantIncome'] > 0).astype(int)
    logger.info("已创建特征: HasCoapplicant (是否有共同申请人)")
    
    # 特征7：收入水平分类（基于总收入的四分位数）
    # 注意：如果只有一个样本，分位数会相同，需要使用固定阈值或去重处理
    if len(df_features) > 1:
        # 多个样本：使用分位数计算
        income_quartiles = df_features['TotalIncome'].quantile([0.25, 0.5, 0.75])
        # 确保分箱边界唯一（去重）
        bins = [0]
        for q in [income_quartiles[0.25], income_quartiles[0.5], income_quartiles[0.75]]:
            if q not in bins:
                bins.append(q)
        bins.append(float('inf'))
        
        # 如果去重后边界不足，使用固定阈值
        if len(bins) < 4:
            # 使用固定阈值（基于常见收入分布）
            bins = [0, 3000, 6000, 10000, float('inf')]
        
        df_features['IncomeLevel'] = pd.cut(
            df_features['TotalIncome'],
            bins=bins,
            labels=['Low', 'Medium', 'High', 'VeryHigh'],
            duplicates='drop'  # 自动去除重复边界
        )
    else:
        # 单个样本（预测时）：使用固定阈值判断
        total_income = df_features['TotalIncome'].iloc[0]
        if total_income < 3000:
            df_features['IncomeLevel'] = 'Low'
        elif total_income < 6000:
            df_features['IncomeLevel'] = 'Medium'
        elif total_income < 10000:
            df_features['IncomeLevel'] = 'High'
        else:
            df_features['IncomeLevel'] = 'VeryHigh'
    logger.info("已创建特征: IncomeLevel (收入水平分类)")
    
    # 特征8：贷款金额分类（基于贷款金额的四分位数）
    # 注意：预测时LoanAmount不存在，使用默认值'Medium'
    if has_loan_amount and df_features['LoanAmount'].notna().sum() > 0:
        loan_quartiles = df_features['LoanAmount'].quantile([0.25, 0.5, 0.75])
        df_features['LoanAmountLevel'] = pd.cut(
            df_features['LoanAmount'],
            bins=[0, loan_quartiles[0.25], loan_quartiles[0.5], 
                  loan_quartiles[0.75], float('inf')],
            labels=['Small', 'Medium', 'Large', 'VeryLarge']
        )
        logger.info("已创建特征: LoanAmountLevel (贷款金额分类)")
    else:
        # 预测时LoanAmount不存在，使用默认值'Medium'
        df_features['LoanAmountLevel'] = 'Medium'
        logger.info("已创建特征: LoanAmountLevel (贷款金额分类，使用默认值Medium)")
    
    return df_features


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    编码分类特征（Label Encoding）
    
    参数:
        df: 包含分类特征的DataFrame
    
    返回:
        编码后的DataFrame和编码器字典
    """
    df_encoded = df.copy()
    encoders = {}
    
    # 需要编码的分类特征
    categorical_features = [
        'Gender', 'Married', 'Dependents', 'Education', 
        'Self_Employed', 'Property_Area', 'IncomeLevel', 'LoanAmountLevel'
    ]
    
    for feature in categorical_features:
        if feature in df_encoded.columns:
            # 处理缺失值（编码前）
            if df_encoded[feature].isnull().sum() > 0:
                df_encoded[feature].fillna('Unknown', inplace=True)
            
            # 创建LabelEncoder
            le = LabelEncoder()
            df_encoded[f'{feature}_encoded'] = le.fit_transform(df_encoded[feature].astype(str))
            
            # 保存编码器（用于后续预测时的特征编码）
            encoders[feature] = le
            
            logger.info(f"已编码特征: {feature} -> {feature}_encoded")
    
    return df_encoded, encoders


def select_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    选择用于模型训练的特征
    
    参数:
        df: 包含所有特征的DataFrame
    
    返回:
        只包含训练特征的DataFrame
    """
    # 基础特征（数值型）
    base_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History'
    ]
    
    # 衍生特征（数值型）
    derived_features = [
        'TotalIncome',
        'LoanToIncomeRatio',
        'MonthlyIncome',
        'MonthlyPayment',
        'PaymentToIncomeRatio',
        'HasCoapplicant'
    ]
    
    # 编码后的分类特征
    encoded_features = [
        'Gender_encoded',
        'Married_encoded',
        'Dependents_encoded',
        'Education_encoded',
        'Self_Employed_encoded',
        'Property_Area_encoded',
        'IncomeLevel_encoded',
        'LoanAmountLevel_encoded'
    ]
    
    # 合并所有特征
    all_features = base_features + derived_features + encoded_features
    
    # 只选择存在的特征
    available_features = [f for f in all_features if f in df.columns]
    
    # 创建特征DataFrame
    feature_df = df[available_features].copy()
    
    logger.info(f"已选择 {len(available_features)} 个特征用于训练")
    logger.info(f"特征列表: {available_features}")
    
    return feature_df


def extract_features_from_database(
    connection,
    limit: Optional[int] = None,
    include_label: bool = True
) -> tuple:
    """
    从数据库提取并处理特征
    
    参数:
        connection: 数据库连接对象
        limit: 限制查询的记录数
        include_label: 是否包含标签（Loan_Status）
    
    返回:
        (特征DataFrame, 标签Series, 编码器字典)
    """
    logger.info("=" * 60)
    logger.info("开始特征提取流程")
    logger.info("=" * 60)
    
    # 步骤1：从数据库查询数据
    logger.info("步骤1：从数据库查询客户数据...")
    df_raw = query_all_customers(connection, limit=limit)
    
    if df_raw.empty:
        logger.error("未查询到数据，特征提取终止")
        return pd.DataFrame(), pd.Series(), {}
    
    logger.info(f"查询到 {len(df_raw)} 条原始记录")
    
    # 步骤2：处理缺失值
    logger.info("步骤2：处理缺失值...")
    df_processed = handle_missing_values(df_raw)
    
    # 步骤3：创建衍生特征
    logger.info("步骤3：创建衍生特征...")
    df_features = create_derived_features(df_processed)
    
    # 步骤4：编码分类特征
    logger.info("步骤4：编码分类特征...")
    df_encoded, encoders = encode_categorical_features(df_features)
    
    # 步骤5：选择训练特征
    logger.info("步骤5：选择训练特征...")
    X = select_features_for_training(df_encoded)
    
    # 步骤6：提取标签（如果需要）
    y = pd.Series()
    if include_label and 'Loan_Status' in df_encoded.columns:
        # 将Loan_Status转换为二值标签：Y->1, N->0
        y = (df_encoded['Loan_Status'] == 'Y').astype(int)
        logger.info(f"已提取标签，正样本数: {y.sum()}, 负样本数: {(y == 0).sum()}")
    
    logger.info("=" * 60)
    logger.info("特征提取完成")
    logger.info("=" * 60)
    
    return X, y, encoders


def save_features(X: pd.DataFrame, y: pd.Series, output_dir: str):
    """
    保存特征数据到文件
    
    参数:
        X: 特征DataFrame
        y: 标签Series
        output_dir: 输出目录路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存特征
    features_path = output_path / 'features.csv'
    X.to_csv(features_path, index=False, encoding='utf-8-sig')
    logger.info(f"特征已保存到: {features_path}")
    
    # 保存标签（如果有）
    if not y.empty:
        labels_path = output_path / 'labels.csv'
        y.to_csv(labels_path, index=False, encoding='utf-8-sig', header=['Loan_Status'])
        logger.info(f"标签已保存到: {labels_path}")


def main():
    """
    主函数：演示特征提取功能
    """
    logger.info("=" * 60)
    logger.info("开始特征提取流程")
    logger.info("=" * 60)
    
    # 步骤1：连接数据库
    logger.info("步骤1：连接数据库...")
    connection = connect_database()
    
    if not connection:
        logger.error("数据库连接失败，特征提取终止")
        return
    
    try:
        # 步骤2：提取特征
        logger.info("步骤2：提取特征...")
        X, y, encoders = extract_features_from_database(
            connection,
            limit=1000,  # 限制查询1000条记录用于演示
            include_label=True
        )
        
        if not X.empty:
            logger.info(f"特征提取成功，特征形状: {X.shape}")
            logger.info(f"特征列: {list(X.columns)}")
            logger.info(f"\n特征预览（前5行）:\n{X.head()}")
            
            if not y.empty:
                logger.info(f"\n标签统计:\n{y.value_counts()}")
            
            # 步骤3：保存特征
            logger.info("\n步骤3：保存特征到文件...")
            output_dir = project_root / 'data' / 'prediction'
            save_features(X, y, str(output_dir))
            
            logger.info(f"\n编码器数量: {len(encoders)}")
            for feature, encoder in encoders.items():
                logger.info(f"  - {feature}: {len(encoder.classes_)} 个类别")
        else:
            logger.warning("特征提取失败，未生成特征数据")
        
    finally:
        # 步骤4：关闭数据库连接
        connection.close()
        logger.info("数据库连接已关闭")
    
    logger.info("=" * 60)
    logger.info("特征提取流程完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

