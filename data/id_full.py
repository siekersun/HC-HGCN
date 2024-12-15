import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('pretrain/Vclinical.csv')

# 假设病人 ID 在 'patient_id' 列，将其转换为 9 位格式
df['Patient ID'] = df['Patient ID'].apply(lambda x: f"{int(x):09}")

# 保存到新的 CSV 文件
df.to_csv('pretrain/Vclinical.csv', index=False)
