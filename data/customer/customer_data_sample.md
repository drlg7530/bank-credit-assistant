# 客户数据样本（前5行）

## 数据说明
本文档包含从 `train.csv` 文件中读取的前5行客户数据样本。

## 数据表结构
| 字段名 | 说明 | 示例值 |
|--------|------|--------|
| Loan_ID | 贷款ID | LP001002 |
| Gender | 性别 | Male |
| Married | 是否已婚 | No |
| Dependents | 家属人数 | 0 |
| Education | 教育程度 | Graduate |
| Self_Employed | 是否自雇 | No |
| ApplicantIncome | 申请人收入 | 5849 |
| CoapplicantIncome | 共同申请人收入 | 0 |
| LoanAmount | 贷款金额 | 空值 |
| Loan_Amount_Term | 贷款期限（天） | 360 |
| Credit_History | 信用历史 | 1 |
| Property_Area | 房产区域 | Urban |
| Loan_Status | 贷款状态 | Y |

## 数据内容

### 第1行（表头）
```
Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status
```

### 第2行
```
LP001002,Male,No,0,Graduate,No,5849,0,,360,1,Urban,Y
```

**解析：**
- Loan_ID: LP001002
- Gender: Male
- Married: No
- Dependents: 0
- Education: Graduate
- Self_Employed: No
- ApplicantIncome: 5849
- CoapplicantIncome: 0
- LoanAmount: （空值）
- Loan_Amount_Term: 360
- Credit_History: 1
- Property_Area: Urban
- Loan_Status: Y

### 第3行
```
LP001003,Male,Yes,1,Graduate,No,4583,1508,128,360,1,Rural,N
```

**解析：**
- Loan_ID: LP001003
- Gender: Male
- Married: Yes
- Dependents: 1
- Education: Graduate
- Self_Employed: No
- ApplicantIncome: 4583
- CoapplicantIncome: 1508
- LoanAmount: 128
- Loan_Amount_Term: 360
- Credit_History: 1
- Property_Area: Rural
- Loan_Status: N

### 第4行
```
LP001005,Male,Yes,0,Graduate,Yes,3000,0,66,360,1,Urban,Y
```

**解析：**
- Loan_ID: LP001005
- Gender: Male
- Married: Yes
- Dependents: 0
- Education: Graduate
- Self_Employed: Yes
- ApplicantIncome: 3000
- CoapplicantIncome: 0
- LoanAmount: 66
- Loan_Amount_Term: 360
- Credit_History: 1
- Property_Area: Urban
- Loan_Status: Y

### 第5行
```
LP001006,Male,Yes,0,Not Graduate,No,2583,2358,120,360,1,Urban,Y
```

**解析：**
- Loan_ID: LP001006
- Gender: Male
- Married: Yes
- Dependents: 0
- Education: Not Graduate
- Self_Employed: No
- ApplicantIncome: 2583
- CoapplicantIncome: 2358
- LoanAmount: 120
- Loan_Amount_Term: 360
- Credit_History: 1
- Property_Area: Urban
- Loan_Status: Y

## 数据统计
- 总记录数：5行（包含表头）
- 数据记录数：4条
- 字段数：13个




