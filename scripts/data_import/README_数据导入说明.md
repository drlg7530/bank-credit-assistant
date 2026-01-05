# 客户数据导入说明

## 功能说明
本脚本用于将 `data/customer/train.csv` 文件中的客户数据导入到MySQL数据库中。

## 使用前准备

### 1. 安装依赖库
```bash
pip install pymysql
```

### 2. 创建数据库和表
确保已经：
- 创建了数据库 `bank_credit_agent`
- 执行了 `data/customer/create_table.sql` 中的建表语句

### 3. 检查数据库配置
确认 `config/database.py` 中的数据库连接配置正确：
- 主机：localhost
- 端口：3306
- 用户：root
- 密码：root
- 数据库名：bank_credit_agent

## 使用方法

### 方式1：直接运行脚本
```bash
python scripts/data_import/import_customer_data.py
```

### 方式2：在项目根目录运行
```bash
cd D:\LLMproject\cursor\myProjects\bank_credit_agent
python scripts/data_import/import_customer_data.py
```

## 功能特点

1. **数据清洗**：自动处理空值和特殊字符
2. **类型转换**：自动将字符串转换为对应的数据类型
3. **批量插入**：使用批量插入提高效率（默认每批100条）
4. **错误处理**：遇到错误记录会跳过并继续处理其他记录
5. **重复处理**：使用 `ON DUPLICATE KEY UPDATE` 处理重复数据
6. **日志记录**：详细的操作日志，保存到 `data_import.log` 文件

## 数据字段映射

| CSV字段 | 数据库字段 | 类型 | 说明 |
|---------|-----------|------|------|
| Loan_ID | Loan_ID | VARCHAR(20) | 贷款ID（主键） |
| Gender | Gender | VARCHAR(10) | 性别 |
| Married | Married | VARCHAR(10) | 是否已婚 |
| Dependents | Dependents | VARCHAR(10) | 家属人数 |
| Education | Education | VARCHAR(20) | 教育程度 |
| Self_Employed | Self_Employed | VARCHAR(10) | 是否自雇 |
| ApplicantIncome | ApplicantIncome | DECIMAL(10,2) | 申请人收入 |
| CoapplicantIncome | CoapplicantIncome | DECIMAL(10,2) | 共同申请人收入 |
| LoanAmount | LoanAmount | DECIMAL(10,2) | 贷款金额（可为空） |
| Loan_Amount_Term | Loan_Amount_Term | INT | 贷款期限（天） |
| Credit_History | Credit_History | TINYINT | 信用历史 |
| Property_Area | Property_Area | VARCHAR(20) | 房产区域 |
| Loan_Status | Loan_Status | VARCHAR(1) | 贷款状态 |

## 注意事项

1. **空值处理**：LoanAmount字段可能为空，脚本会自动处理为NULL
2. **特殊值处理**：Dependents字段可能包含"3+"这样的值，会原样保存
3. **重复数据**：如果Loan_ID已存在，会更新现有记录而不是报错
4. **事务处理**：每批数据作为一个事务，失败会回滚该批数据

## 日志文件

脚本运行时会生成 `data_import.log` 日志文件，包含：
- 数据读取进度
- 插入成功/失败记录
- 错误详细信息

## 常见问题

### 1. 连接数据库失败
- 检查MySQL服务是否启动
- 检查数据库配置是否正确
- 检查数据库用户权限

### 2. 表不存在错误
- 确保已执行建表SQL语句
- 检查表名是否正确

### 3. 编码问题
- 确保CSV文件使用UTF-8编码
- 数据库使用utf8mb4字符集




