"""
LightGBM模型预测测试模块
功能：使用训练好的模型预测客户未来n个月的贷款金额（测试脚本）

使用说明：
1. 确保已训练并保存了模型（运行model_train.py）
2. 运行预测测试：python src/prediction/model_predict_test.py

注意：
- 这是一个测试脚本，用于验证预测功能
- 实际系统集成时，应使用模块中的函数而非直接运行此脚本
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import lightgbm as lgb

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入特征提取模块
from src.prediction.feature_extraction import (
    handle_missing_values,
    create_derived_features,
    encode_categorical_features,
    select_features_for_training
)

# 导入数据查询模块
from src.prediction.data_query import connect_database, query_all_customers

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_predict_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model(model_dir: str) -> tuple:
    """
    加载训练好的模型和相关文件
    
    参数:
        model_dir: 模型目录路径
    
    返回:
        (模型, 编码器字典, 特征名称列表, 评估指标字典)
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    # 加载模型
    model_file = model_path / 'lightgbm_model.txt'
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    model = lgb.Booster(model_file=str(model_file))
    logger.info(f"模型已加载: {model_file}")
    
    # 加载编码器
    encoders_file = model_path / 'encoders.pkl'
    if not encoders_file.exists():
        raise FileNotFoundError(f"编码器文件不存在: {encoders_file}")
    
    with open(encoders_file, 'rb') as f:
        encoders = pickle.load(f)
    logger.info(f"编码器已加载: {encoders_file}")
    
    # 加载特征名称
    feature_names_file = model_path / 'feature_names.pkl'
    if not feature_names_file.exists():
        raise FileNotFoundError(f"特征名称文件不存在: {feature_names_file}")
    
    with open(feature_names_file, 'rb') as f:
        feature_names = pickle.load(f)
    logger.info(f"特征名称已加载: {feature_names_file}")
    
    # 加载评估指标（可选）
    metrics_file = model_path / 'metrics.pkl'
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, 'rb') as f:
            metrics = pickle.load(f)
        logger.info(f"评估指标已加载: {metrics_file}")
    
    return model, encoders, feature_names, metrics


def prepare_customer_features(
    customer_data: Dict[str, Any],
    encoders: Dict,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    准备客户特征用于预测
    
    参数:
        customer_data: 客户数据字典（包含原始字段）
        encoders: 特征编码器字典
        feature_names: 训练时的特征名称列表
    
    返回:
        处理后的特征DataFrame
    """
    logger.info("开始准备客户特征...")
    
    # 步骤1：将客户数据转换为DataFrame
    df_raw = pd.DataFrame([customer_data])
    
    # 步骤2：处理缺失值
    df_processed = handle_missing_values(df_raw)
    
    # 步骤3：创建衍生特征
    df_features = create_derived_features(df_processed)
    
    # 步骤4：编码分类特征
    # 注意：这里需要使用训练时保存的编码器，而不是创建新的编码器
    df_encoded = df_features.copy()
    
    # 使用训练时保存的编码器进行编码
    for feature, encoder in encoders.items():
        if feature in df_encoded.columns:
            # 处理缺失值
            if df_encoded[feature].isnull().any():
                df_encoded[feature].fillna('Unknown', inplace=True)
            
            # 处理未知类别（如果预测时出现训练时没有的类别）
            try:
                # 检查是否有未知类别
                unknown_mask = ~df_encoded[feature].astype(str).isin(encoder.classes_)
                if unknown_mask.any():
                    logger.warning(f"特征 {feature} 出现未知类别，使用最常见的类别")
                    most_common = encoder.classes_[0]
                    df_encoded.loc[unknown_mask, feature] = most_common
                
                # 使用训练时的编码器进行编码
                df_encoded[f'{feature}_encoded'] = encoder.transform(
                    df_encoded[feature].astype(str)
                )
            except Exception as e:
                logger.error(f"特征 {feature} 编码失败: {e}")
                # 使用最常见的类别编码作为默认值
                most_common = encoder.classes_[0]
                df_encoded[f'{feature}_encoded'] = encoder.transform([most_common])[0]
    
    # 步骤5：选择训练特征
    X = select_features_for_training(df_encoded)
    
    # 步骤6：确保特征顺序与训练时一致
    # 如果缺少某些特征，用0填充
    for feature in feature_names:
        if feature not in X.columns:
            logger.warning(f"特征 {feature} 缺失，使用0填充")
            X[feature] = 0
    
    # 按训练时的特征顺序排列
    X = X[feature_names]
    
    logger.info(f"特征准备完成，特征形状: {X.shape}")
    
    return X


def predict_loan_amount(
    customer_data: Dict[str, Any],
    model_dir: Optional[str] = None,
    months_ahead: int = 1
) -> float:
    """
    预测客户未来n个月的贷款金额
    
    参数:
        customer_data: 客户数据字典，包含以下字段：
            - Gender: 性别 (Male/Female)
            - Married: 是否已婚 (Yes/No)
            - Dependents: 家属人数 (0/1/2/3+)
            - Education: 教育程度 (Graduate/Not Graduate)
            - Self_Employed: 是否自雇 (Yes/No)
            - ApplicantIncome: 申请人收入
            - CoapplicantIncome: 共同申请人收入
            - Loan_Amount_Term: 贷款期限（天）
            - Credit_History: 信用历史 (0/1)
            - Property_Area: 房产区域 (Urban/Rural/Semiurban)
        model_dir: 模型目录路径（默认使用models/lightgbm）
        months_ahead: 预测未来几个月（当前实现为基础预测，未来可扩展时间序列预测）
    
    返回:
        预测的贷款金额
    """
    # 默认模型目录
    if model_dir is None:
        model_dir = project_root / 'models' / 'lightgbm'
    
    # 加载模型
    logger.info("=" * 60)
    logger.info("开始预测流程")
    logger.info("=" * 60)
    logger.info(f"加载模型: {model_dir}")
    
    model, encoders, feature_names, metrics = load_model(str(model_dir))
    
    # 准备特征
    logger.info("准备客户特征...")
    X = prepare_customer_features(customer_data, encoders, feature_names)
    
    # 预测
    logger.info("进行预测...")
    prediction = model.predict(X, num_iteration=model.best_iteration)[0]
    
    # 如果预测值为负数，设为0（贷款金额不能为负）
    prediction = max(0, prediction)
    
    logger.info(f"预测完成，未来{months_ahead}个月预测贷款金额: {prediction:.2f} 元")
    logger.info("=" * 60)
    
    return prediction


def predict_batch(
    customers_data: List[Dict[str, Any]],
    model_dir: Optional[str] = None,
    months_ahead: int = 1
) -> List[float]:
    """
    批量预测多个客户的贷款金额
    
    参数:
        customers_data: 客户数据列表
        model_dir: 模型目录路径
        months_ahead: 预测未来几个月
    
    返回:
        预测结果列表
    """
    # 默认模型目录
    if model_dir is None:
        model_dir = project_root / 'models' / 'lightgbm'
    
    # 加载模型
    logger.info("=" * 60)
    logger.info("开始批量预测")
    logger.info("=" * 60)
    logger.info(f"加载模型: {model_dir}")
    
    model, encoders, feature_names, metrics = load_model(str(model_dir))
    
    # 准备所有客户的特征
    X_list = []
    for i, customer_data in enumerate(customers_data):
        logger.info(f"处理客户 {i+1}/{len(customers_data)}...")
        X = prepare_customer_features(customer_data, encoders, feature_names)
        X_list.append(X)
    
    # 合并所有特征
    X_all = pd.concat(X_list, ignore_index=True)
    
    # 批量预测
    logger.info("进行批量预测...")
    predictions = model.predict(X_all, num_iteration=model.best_iteration)
    
    # 处理负值
    predictions = np.maximum(predictions, 0)
    
    logger.info(f"批量预测完成，共预测 {len(predictions)} 个客户")
    logger.info("=" * 60)
    
    return predictions.tolist()


def save_predictions_to_csv(
    predictions_df: pd.DataFrame,
    output_path: str
) -> bool:
    """
    保存预测结果到CSV文件
    
    参数:
        predictions_df: 包含预测结果的DataFrame
        output_path: 输出文件路径
    
    返回:
        保存是否成功
    """
    try:
        # 确保输出目录存在
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"预测结果已保存到: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")
        return False


def main():
    """
    主函数：预测数据库中所有客户未来一个月的贷款金额，并保存到CSV
    """
    logger.info("=" * 60)
    logger.info("开始LightGBM模型预测测试")
    logger.info("预测数据库中所有客户未来一个月的贷款金额")
    logger.info("=" * 60)
    
    # 检查模型是否存在
    model_dir = project_root / 'models' / 'lightgbm'
    if not (model_dir / 'lightgbm_model.txt').exists():
        logger.error(f"模型文件不存在，请先运行 model_train.py 训练模型")
        logger.error(f"模型路径: {model_dir}")
        return
    
    # 步骤1：连接数据库
    logger.info("\n步骤1：连接数据库...")
    connection = connect_database()
    if not connection:
        logger.error("数据库连接失败，预测终止")
        return
    
    try:
        # 步骤2：查询所有客户数据
        logger.info("步骤2：查询所有客户数据...")
        df_customers = query_all_customers(connection, limit=None)
        
        if df_customers.empty:
            logger.error("未查询到客户数据，预测终止")
            return
        
        logger.info(f"成功查询到 {len(df_customers)} 条客户记录")
        
        # 步骤3：将DataFrame转换为字典列表（用于预测）
        logger.info("步骤3：准备客户数据...")
        customers_data = []
        loan_ids = []
        
        for _, row in df_customers.iterrows():
            # 处理数值字段的NaN值
            applicant_income = row.get('ApplicantIncome')
            applicant_income = float(applicant_income) if pd.notna(applicant_income) else 0.0
            
            coapplicant_income = row.get('CoapplicantIncome')
            coapplicant_income = float(coapplicant_income) if pd.notna(coapplicant_income) else 0.0
            
            loan_amount_term = row.get('Loan_Amount_Term')
            loan_amount_term = int(loan_amount_term) if pd.notna(loan_amount_term) else 360
            
            credit_history = row.get('Credit_History')
            credit_history = int(credit_history) if pd.notna(credit_history) else 0
            
            customer_dict = {
                'Gender': row.get('Gender') if pd.notna(row.get('Gender')) else 'Unknown',
                'Married': row.get('Married') if pd.notna(row.get('Married')) else 'Unknown',
                'Dependents': row.get('Dependents') if pd.notna(row.get('Dependents')) else '0',
                'Education': row.get('Education') if pd.notna(row.get('Education')) else 'Unknown',
                'Self_Employed': row.get('Self_Employed') if pd.notna(row.get('Self_Employed')) else 'Unknown',
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': row.get('Property_Area') if pd.notna(row.get('Property_Area')) else 'Unknown'
            }
            customers_data.append(customer_dict)
            loan_ids.append(row.get('Loan_ID'))
        
        logger.info(f"已准备 {len(customers_data)} 个客户的数据")
        
        # 步骤4：批量预测所有客户
        logger.info("\n步骤4：开始批量预测...")
        logger.info(f"预测未来1个月的贷款金额...")
        
        predictions = predict_batch(
            customers_data,
            model_dir=str(model_dir),
            months_ahead=1  # 预测未来1个月
        )
        
        logger.info(f"预测完成，共预测 {len(predictions)} 个客户")
        
        # 步骤5：构建结果DataFrame
        logger.info("\n步骤5：整理预测结果...")
        results_df = pd.DataFrame({
            'Loan_ID': loan_ids,
            'PredictedLoanAmount_1Month': predictions,
            'Gender': [c.get('Gender') for c in customers_data],
            'Married': [c.get('Married') for c in customers_data],
            'ApplicantIncome': [c.get('ApplicantIncome') for c in customers_data],
            'CoapplicantIncome': [c.get('CoapplicantIncome') for c in customers_data],
            'TotalIncome': [c.get('ApplicantIncome', 0) + c.get('CoapplicantIncome', 0) for c in customers_data],
            'Credit_History': [c.get('Credit_History') for c in customers_data],
            'Property_Area': [c.get('Property_Area') for c in customers_data]
        })
        
        # 步骤6：保存结果到CSV
        logger.info("\n步骤6：保存预测结果到CSV文件...")
        output_path = project_root / 'data' / 'prediction' / 'predicted_loan_amounts_1month.csv'
        if save_predictions_to_csv(results_df, str(output_path)):
            logger.info(f"成功保存 {len(results_df)} 条预测结果")
            
            # 显示统计信息
            logger.info("\n预测结果统计:")
            logger.info(f"  总客户数: {len(results_df)}")
            logger.info(f"  平均预测贷款金额: {results_df['PredictedLoanAmount_1Month'].mean():.2f} 元")
            logger.info(f"  最小预测贷款金额: {results_df['PredictedLoanAmount_1Month'].min():.2f} 元")
            logger.info(f"  最大预测贷款金额: {results_df['PredictedLoanAmount_1Month'].max():.2f} 元")
            logger.info(f"  中位数预测贷款金额: {results_df['PredictedLoanAmount_1Month'].median():.2f} 元")
            
            # 显示前10条结果
            logger.info("\n前10条预测结果:")
            logger.info(f"\n{results_df.head(10).to_string(index=False)}")
        else:
            logger.error("保存预测结果失败")
        
    except Exception as e:
        logger.error(f"预测过程发生错误: {e}", exc_info=True)
    
    finally:
        # 关闭数据库连接
        if connection:
            connection.close()
            logger.info("\n数据库连接已关闭")
    
    logger.info("=" * 60)
    logger.info("预测测试完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

