### 環境
env.yaml

### 資料集
1. 預訓練：newVer_wang_summary.py
2. 微調：newVer_sighan_summary.py
3. 驗證：sighan13/14/15_summary.json

### 步驟
*需先下載correction bert 的預訓練模型：https://drive.google.com/file/d/1CseJzc58W4s8U_eIuAnshHQmnmi7Sr5-/view?usp=sharing \n
**將資料夾放在根目錄下即可

1. 執行 start.py 進行預訓練
2. 載入預訓練模型(已寫在腳本中)，執行 sighan_finetuned.py
3. 執行 eval.py 得到 p, r, f 指標結果
