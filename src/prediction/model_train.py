"""
LightGBM模型训练模块
功能：训练LightGBM回归模型，预测客户未来n个月的贷款金额

使用说明：
1. 确保已安装lightgbm、scikit-learn
2. 运行训练：python src/prediction/model_train.py
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入特征提取模块
from src.prediction.data_query import connect_database
from src.prediction.feature_extraction import (
    extract_features_from_database,
    handle_missing_values,
    create_derived_features,
    encode_categorical_features,
    select_features_for_training
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_train.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_features_for_loan_amount(
    connection,
    limit: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    提取用于贷款金额预测的特征和标签
    
    参数:
        connection: 数据库连接对象
        limit: 限制查询的记录数
    
    返回:
        (特征DataFrame, 标签Series, 编码器字典)
    """
    logger.info("=" * 60)
    logger.info("开始提取贷款金额预测特征")
    logger.info("=" * 60)
    
    # 步骤1：从数据库查询数据
    logger.info("步骤1：从数据库查询客户数据...")
    from src.prediction.data_query import query_all_customers
    df_raw = query_all_customers(connection, limit=limit)
    
    if df_raw.empty:
        logger.error("未查询到数据，特征提取终止")
        return pd.DataFrame(), pd.Series(), {}
    
    logger.info(f"查询到 {len(df_raw)} 条原始记录")
    
    # 步骤2：过滤掉LoanAmount为空的记录（回归问题需要标签）
    df_raw = df_raw[df_raw['LoanAmount'].notna()].copy()
    logger.info(f"过滤后剩余 {len(df_raw)} 条有效记录（LoanAmount不为空）")
    
    if df_raw.empty:
        logger.error("没有有效的标签数据，特征提取终止")
        return pd.DataFrame(), pd.Series(), {}
    
    # 步骤3：处理缺失值
    logger.info("步骤3：处理缺失值...")
    df_processed = handle_missing_values(df_raw)
    
    # 步骤4：创建衍生特征
    logger.info("步骤4：创建衍生特征...")
    df_features = create_derived_features(df_processed)
    
    # 步骤5：编码分类特征
    logger.info("步骤5：编码分类特征...")
    df_encoded, encoders = encode_categorical_features(df_features)
    
    # 步骤6：选择训练特征
    logger.info("步骤6：选择训练特征...")
    X = select_features_for_training(df_encoded)
    
    # 步骤7：提取标签（LoanAmount作为回归目标）
    y = df_encoded['LoanAmount'].copy()
    logger.info(f"已提取标签，贷款金额范围: {y.min():.2f} ~ {y.max():.2f}, 平均值: {y.mean():.2f}")
    
    logger.info("=" * 60)
    logger.info("特征提取完成")
    logger.info("=" * 60)
    
    return X, y, encoders


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> lgb.Booster:
    """
    训练LightGBM回归模型
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        params: LightGBM参数（可选）
    
    返回:
        训练好的LightGBM模型
    """
    logger.info("=" * 60)
    logger.info("开始训练LightGBM模型")
    logger.info("=" * 60)
    
    # 默认参数配置（回归问题）
    default_params = {
        'objective': 'regression',           # 回归任务
        'metric': 'rmse',                    # 评估指标：均方根误差
        'boosting_type': 'gbdt',            # 梯度提升决策树
        'num_leaves': 31,                   # 叶子节点数
        'learning_rate': 0.05,               # 学习率
        'feature_fraction': 0.9,            # 特征采样比例
        'bagging_fraction': 0.8,            # 数据采样比例
        'bagging_freq': 5,                  # 数据采样频率
        'verbose': -1,                      # 不输出详细信息
        'random_state': 42,                  # 随机种子
    }
    
    # 合并用户参数
    if params:
        default_params.update(params)
    
    logger.info(f"模型参数: {default_params}")
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 训练模型
    logger.info("开始训练...")
    model = lgb.train(
        default_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,                # 迭代轮数
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # 早停：50轮无改善则停止
            lgb.log_evaluation(period=50)            # 每50轮输出一次评估结果
        ]
    )
    
    logger.info("模型训练完成")
    logger.info("=" * 60)
    
    return model


def evaluate_model(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "数据集"
) -> Dict[str, float]:
    """
    评估模型性能
    
    参数:
        model: 训练好的LightGBM模型
        X: 特征数据
        y: 真实标签
        dataset_name: 数据集名称（用于日志）
    
    返回:
        包含评估指标的字典
    """
    # 预测
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    
    # 计算评估指标
    mae = mean_absolute_error(y, y_pred)           # 平均绝对误差
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # 均方根误差
    r2 = r2_score(y, y_pred)                      # R²决定系数
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    logger.info(f"{dataset_name}评估结果:")
    logger.info(f"  MAE (平均绝对误差): {mae:.2f}")
    logger.info(f"  RMSE (均方根误差): {rmse:.2f}")
    logger.info(f"  R² (决定系数): {r2:.4f}")
    
    return metrics


def save_model(
    model: lgb.Booster,
    encoders: Dict,
    feature_names: list,
    metrics: Dict[str, Dict[str, float]],
    output_dir: str
):
    """
    保存训练好的模型和相关文件
    
    参数:
        model: 训练好的LightGBM模型
        encoders: 特征编码器字典
        feature_names: 特征名称列表
        metrics: 评估指标字典
        output_dir: 输出目录路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_path / 'lightgbm_model.txt'
    model.save_model(str(model_path))
    logger.info(f"模型已保存到: {model_path}")
    
    # 保存编码器
    encoders_path = output_path / 'encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    logger.info(f"编码器已保存到: {encoders_path}")
    
    # 保存特征名称
    feature_names_path = output_path / 'feature_names.pkl'
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f"特征名称已保存到: {feature_names_path}")
    
    # 保存评估指标
    metrics_path = output_path / 'metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    logger.info(f"评估指标已保存到: {metrics_path}")
    
    # 保存特征重要性
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    importance_path = output_path / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
    logger.info(f"特征重要性已保存到: {importance_path}")
    logger.info(f"\n前10个重要特征:\n{importance_df.head(10)}")


def train_loan_amount_model(
    connection,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    limit: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[lgb.Booster, Dict, list, Dict[str, Dict[str, float]]]:
    """
    完整的模型训练流程
    
    参数:
        connection: 数据库连接对象
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中划分）
        random_state: 随机种子
        limit: 限制查询的记录数
        params: LightGBM参数（可选）
    
    返回:
        (模型, 编码器字典, 特征名称列表, 评估指标字典)
    """
    logger.info("=" * 60)
    logger.info("开始模型训练流程")
    logger.info("=" * 60)
    
    # 步骤1：提取特征和标签
    logger.info("步骤1：提取特征和标签...")
    X, y, encoders = extract_features_for_loan_amount(connection, limit=limit)
    
    if X.empty or y.empty:
        logger.error("特征提取失败，训练终止")
        return None, {}, [], {}
    
    # 保存特征名称
    feature_names = list(X.columns)
    logger.info(f"特征数量: {len(feature_names)}")
    
    # 步骤2：划分数据集
    logger.info("步骤2：划分数据集...")
    # 先划分出测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 再从训练集中划分出验证集
    val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    logger.info(f"训练集: {len(X_train)} 条")
    logger.info(f"验证集: {len(X_val)} 条")
    logger.info(f"测试集: {len(X_test)} 条")
    
    # 步骤3：训练模型
    logger.info("步骤3：训练模型...")
    model = train_lightgbm_model(X_train, y_train, X_val, y_val, params=params)
    
    # 步骤4：评估模型
    logger.info("步骤4：评估模型...")
    train_metrics = evaluate_model(model, X_train, y_train, "训练集")
    val_metrics = evaluate_model(model, X_val, y_val, "验证集")
    test_metrics = evaluate_model(model, X_test, y_test, "测试集")
    
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    # 步骤5：保存模型
    logger.info("步骤5：保存模型...")
    output_dir = project_root / 'models' / 'lightgbm'
    save_model(model, encoders, feature_names, metrics, str(output_dir))
    
    logger.info("=" * 60)
    logger.info("模型训练流程完成")
    logger.info("=" * 60)
    
    return model, encoders, feature_names, metrics


def main():
    """
    主函数：演示模型训练功能
    """
    logger.info("=" * 60)
    logger.info("开始LightGBM模型训练")
    logger.info("=" * 60)
    
    # 步骤1：连接数据库
    logger.info("步骤1：连接数据库...")
    connection = connect_database()
    
    if not connection:
        logger.error("数据库连接失败，训练终止")
        return
    
    try:
        # 步骤2：训练模型
        logger.info("步骤2：开始训练模型...")
        model, encoders, feature_names, metrics = train_loan_amount_model(
            connection,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            limit=1000,  # 限制查询1000条记录用于演示
            params=None  # 使用默认参数
        )
        
        if model:
            logger.info("\n训练成功！")
            logger.info(f"特征数量: {len(feature_names)}")
            logger.info(f"\n测试集性能:")
            logger.info(f"  MAE: {metrics['test']['MAE']:.2f}")
            logger.info(f"  RMSE: {metrics['test']['RMSE']:.2f}")
            logger.info(f"  R²: {metrics['test']['R2']:.4f}")
        else:
            logger.error("模型训练失败")
        
    finally:
        # 步骤3：关闭数据库连接
        connection.close()
        logger.info("数据库连接已关闭")
    
    logger.info("=" * 60)
    logger.info("模型训练流程完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

