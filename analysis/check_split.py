"""诊断脚本：检查各 split mode 的 test 集是否有效。"""
import sys, os, pandas as pd
sys.path.insert(0, '.')
import Config, Data_Process
params = Config.parse()

# 构造 DrugToID（train_valid_test_split 需要）
path = f"{params.data_file}/ONEIL/ONEIL_DRUG.csv"
Drug_Information = pd.read_csv(path).drop_duplicates(subset=['PubChem_CID'])
Drug_Information['Drug_ID'] = range(len(Drug_Information))
DrugToID = Drug_Information[['Name','PubChem_CID','Drug_ID']]
numDrug = len(DrugToID)

for mode in ['random', 'lco', 'ldo', 'lbo']:
    Data_Process.params.split_mode = mode
    print(f"\n{'='*60}\nSplit: {mode}\n{'='*60}")
    try:
        realFolds, *_ = Data_Process.train_valid_test_split('ONEIL', DrugToID, numDrug)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    for iFold, fold in realFolds.items():
        total_test = sum(len(lbl) for lbl in fold.test_labels.values())
        pos_test = sum(1 for lbl in fold.test_labels.values() for l in lbl if l == 1)
        total_train = sum(len(lbl) for lbl in fold.train_labels.values())
        pos_train = sum(1 for lbl in fold.train_labels.values() for l in lbl if l == 1)

        print(f"  Fold {iFold}: train={total_train}(pos={pos_train}) test={total_test}(pos={pos_test})")
        if pos_test == 0:
            print(f"    ⚠️  NO POSITIVE TEST SAMPLES!")
