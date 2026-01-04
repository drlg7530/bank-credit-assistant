"""
预测模块
包含客户贷款金额预测相关的功能模块
"""

from src.prediction.data_query import (
    connect_database,
    query_all_customers,
    query_customers_by_condition,
    query_customer_statistics,
    save_to_csv
)

from src.prediction.feature_extraction import (
    handle_missing_values,
    create_derived_features,
    encode_categorical_features,
    select_features_for_training,
    extract_features_from_database,
    save_features
)

from src.prediction.model_train import (
    train_loan_amount_model,
    extract_features_for_loan_amount,
    train_lightgbm_model,
    evaluate_model,
    save_model
)

from src.prediction.model_predict_test import (
    load_model,
    prepare_customer_features,
    predict_loan_amount,
    predict_batch
)

__all__ = [
    # 数据查询相关
    'connect_database',
    'query_all_customers',
    'query_customers_by_condition',
    'query_customer_statistics',
    'save_to_csv',
    # 特征提取相关
    'handle_missing_values',
    'create_derived_features',
    'encode_categorical_features',
    'select_features_for_training',
    'extract_features_from_database',
    'save_features',
    # 模型训练相关
    'train_loan_amount_model',
    'extract_features_for_loan_amount',
    'train_lightgbm_model',
    'evaluate_model',
    'save_model',
    # 模型预测相关
    'load_model',
    'prepare_customer_features',
    'predict_loan_amount',
    'predict_batch',
]

