-- ============================================
-- 客户数据表建表语句
-- 基于 train.csv 文件结构创建
-- ============================================

-- MySQL 建表语句
-- 方案1：使用 DATETIME + TIMESTAMP（推荐，兼容性好）
CREATE TABLE IF NOT EXISTS customer_loan_data (
    -- 主键：贷款ID，唯一标识每笔贷款
    Loan_ID VARCHAR(20) PRIMARY KEY COMMENT '贷款ID',
    
    -- 基本信息字段
    Gender VARCHAR(10) COMMENT '性别：Male/Female',
    Married VARCHAR(10) COMMENT '是否已婚：Yes/No',
    Dependents VARCHAR(10) COMMENT '家属人数：0/1/2/3+',
    Education VARCHAR(20) COMMENT '教育程度：Graduate/Not Graduate',
    Self_Employed VARCHAR(10) COMMENT '是否自雇：Yes/No',
    
    -- 收入信息字段
    ApplicantIncome DECIMAL(10, 2) COMMENT '申请人收入（元）',
    CoapplicantIncome DECIMAL(10, 2) DEFAULT 0 COMMENT '共同申请人收入（元）',
    
    -- 贷款信息字段
    LoanAmount DECIMAL(10, 2) COMMENT '贷款金额（元）',
    Loan_Amount_Term INT COMMENT '贷款期限（天）',
    
    -- 信用信息字段
    Credit_History TINYINT COMMENT '信用历史：0-无信用记录，1-有信用记录',
    
    -- 房产信息字段
    Property_Area VARCHAR(20) COMMENT '房产区域：Urban/Rural/Semiurban',
    
    -- 贷款状态字段
    Loan_Status VARCHAR(1) COMMENT '贷款状态：Y-批准，N-拒绝',
    
    -- 创建和更新时间戳
    -- 注意：MySQL 5.6.5之前版本的兼容性处理
    -- created_at 使用 TIMESTAMP（只设置默认值，不自动更新）
    -- updated_at 使用 DATETIME，通过触发器自动更新
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NULL COMMENT '更新时间'
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='客户贷款数据表';

-- ============================================
-- 创建触发器：自动设置和更新 updated_at 字段
-- ============================================

-- 插入时自动设置 updated_at 为当前时间
DELIMITER $$
CREATE TRIGGER customer_loan_data_insert_timestamp 
BEFORE INSERT ON customer_loan_data
FOR EACH ROW
BEGIN
    IF NEW.updated_at IS NULL THEN
        SET NEW.updated_at = NOW();
    END IF;
END$$

-- 更新时自动更新 updated_at 为当前时间
CREATE TRIGGER customer_loan_data_update_timestamp 
BEFORE UPDATE ON customer_loan_data
FOR EACH ROW
BEGIN
    SET NEW.updated_at = NOW();
END$$
DELIMITER ;

-- ============================================
-- MySQL 备选方案（如果上述方案仍有问题，可以使用此方案）
-- 方案2：两个字段都使用 DATETIME，通过触发器自动更新 updated_at
-- ============================================
/*
CREATE TABLE IF NOT EXISTS customer_loan_data (
    -- 主键：贷款ID，唯一标识每笔贷款
    Loan_ID VARCHAR(20) PRIMARY KEY COMMENT '贷款ID',
    
    -- 基本信息字段
    Gender VARCHAR(10) COMMENT '性别：Male/Female',
    Married VARCHAR(10) COMMENT '是否已婚：Yes/No',
    Dependents VARCHAR(10) COMMENT '家属人数：0/1/2/3+',
    Education VARCHAR(20) COMMENT '教育程度：Graduate/Not Graduate',
    Self_Employed VARCHAR(10) COMMENT '是否自雇：Yes/No',
    
    -- 收入信息字段
    ApplicantIncome DECIMAL(10, 2) COMMENT '申请人收入（元）',
    CoapplicantIncome DECIMAL(10, 2) DEFAULT 0 COMMENT '共同申请人收入（元）',
    
    -- 贷款信息字段
    LoanAmount DECIMAL(10, 2) COMMENT '贷款金额（元）',
    Loan_Amount_Term INT COMMENT '贷款期限（天）',
    
    -- 信用信息字段
    Credit_History TINYINT COMMENT '信用历史：0-无信用记录，1-有信用记录',
    
    -- 房产信息字段
    Property_Area VARCHAR(20) COMMENT '房产区域：Urban/Rural/Semiurban',
    
    -- 贷款状态字段
    Loan_Status VARCHAR(1) COMMENT '贷款状态：Y-批准，N-拒绝',
    
    -- 创建和更新时间戳（都使用DATETIME类型）
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '更新时间'
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='客户贷款数据表';

-- 创建触发器，自动更新 updated_at 字段
DELIMITER $$
CREATE TRIGGER update_customer_loan_data_timestamp 
BEFORE UPDATE ON customer_loan_data
FOR EACH ROW
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END$$
DELIMITER ;
*/

-- ============================================
-- PostgreSQL 建表语句
-- ============================================
/*
CREATE TABLE IF NOT EXISTS customer_loan_data (
    -- 主键：贷款ID，唯一标识每笔贷款
    Loan_ID VARCHAR(20) PRIMARY KEY,
    
    -- 基本信息字段
    Gender VARCHAR(10),
    Married VARCHAR(10),
    Dependents VARCHAR(10),
    Education VARCHAR(20),
    Self_Employed VARCHAR(10),
    
    -- 收入信息字段
    ApplicantIncome DECIMAL(10, 2),
    CoapplicantIncome DECIMAL(10, 2) DEFAULT 0,
    
    -- 贷款信息字段
    LoanAmount DECIMAL(10, 2),
    Loan_Amount_Term INTEGER,
    
    -- 信用信息字段
    Credit_History SMALLINT,
    
    -- 房产信息字段
    Property_Area VARCHAR(20),
    
    -- 贷款状态字段
    Loan_Status VARCHAR(1),
    
    -- 创建和更新时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 添加注释
COMMENT ON TABLE customer_loan_data IS '客户贷款数据表';
COMMENT ON COLUMN customer_loan_data.Loan_ID IS '贷款ID';
COMMENT ON COLUMN customer_loan_data.Gender IS '性别：Male/Female';
COMMENT ON COLUMN customer_loan_data.Married IS '是否已婚：Yes/No';
COMMENT ON COLUMN customer_loan_data.Dependents IS '家属人数：0/1/2/3+';
COMMENT ON COLUMN customer_loan_data.Education IS '教育程度：Graduate/Not Graduate';
COMMENT ON COLUMN customer_loan_data.Self_Employed IS '是否自雇：Yes/No';
COMMENT ON COLUMN customer_loan_data.ApplicantIncome IS '申请人收入（元）';
COMMENT ON COLUMN customer_loan_data.CoapplicantIncome IS '共同申请人收入（元）';
COMMENT ON COLUMN customer_loan_data.LoanAmount IS '贷款金额（元）';
COMMENT ON COLUMN customer_loan_data.Loan_Amount_Term IS '贷款期限（天）';
COMMENT ON COLUMN customer_loan_data.Credit_History IS '信用历史：0-无信用记录，1-有信用记录';
COMMENT ON COLUMN customer_loan_data.Property_Area IS '房产区域：Urban/Rural/Semiurban';
COMMENT ON COLUMN customer_loan_data.Loan_Status IS '贷款状态：Y-批准，N-拒绝';
*/

-- ============================================
-- SQLite 建表语句
-- ============================================
/*
CREATE TABLE IF NOT EXISTS customer_loan_data (
    -- 主键：贷款ID，唯一标识每笔贷款
    Loan_ID TEXT PRIMARY KEY,
    
    -- 基本信息字段
    Gender TEXT,
    Married TEXT,
    Dependents TEXT,
    Education TEXT,
    Self_Employed TEXT,
    
    -- 收入信息字段
    ApplicantIncome REAL,
    CoapplicantIncome REAL DEFAULT 0,
    
    -- 贷款信息字段
    LoanAmount REAL,
    Loan_Amount_Term INTEGER,
    
    -- 信用信息字段
    Credit_History INTEGER,
    
    -- 房产信息字段
    Property_Area TEXT,
    
    -- 贷款状态字段
    Loan_Status TEXT,
    
    -- 创建和更新时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
*/

-- ============================================
-- 索引创建语句（可选，用于提高查询性能）
-- ============================================

-- 为常用查询字段创建索引
CREATE INDEX idx_loan_status ON customer_loan_data(Loan_Status);
CREATE INDEX idx_credit_history ON customer_loan_data(Credit_History);
CREATE INDEX idx_property_area ON customer_loan_data(Property_Area);
CREATE INDEX idx_gender ON customer_loan_data(Gender);
CREATE INDEX idx_education ON customer_loan_data(Education);

-- ============================================
-- 字段说明
-- ============================================
/*
字段详细说明：
1. Loan_ID: 贷款唯一标识符，主键
2. Gender: 申请人性别，取值为 Male 或 Female
3. Married: 婚姻状况，取值为 Yes 或 No
4. Dependents: 家属人数，取值为 0, 1, 2, 或 3+
5. Education: 教育程度，取值为 Graduate（毕业）或 Not Graduate（未毕业）
6. Self_Employed: 是否自雇，取值为 Yes 或 No
7. ApplicantIncome: 申请人月收入，单位为元
8. CoapplicantIncome: 共同申请人月收入，单位为元，可能为0
9. LoanAmount: 申请的贷款金额，单位为元，可能为空
10. Loan_Amount_Term: 贷款期限，单位为天，常见值为360（一年）
11. Credit_History: 信用历史记录，0表示无信用记录，1表示有信用记录
12. Property_Area: 房产所在区域，取值为 Urban（城市）、Rural（农村）或 Semiurban（半城市）
13. Loan_Status: 贷款审批状态，Y表示批准，N表示拒绝
*/

