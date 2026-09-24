"""提取各细胞系预测协同分最高的药物对。"""
import torch, numpy as np, pandas as pd, sys, os
sys.path.insert(0, '.')
import Config, Data_Process, Synergy_Models

args = Config.parse()
dataset = args.dataset
device = args.device

ckpt = f"checkpoints/{dataset}_pretrained.pt"
if not os.path.exists(ckpt):
    print(f"No checkpoint: {ckpt}"); sys.exit(1)

(realFolds, _, _, _, numNode1, Drug_Features, Cell_Line_Feature,
 DrugToID, _, CellLineToID, V1, E1, edge_num1, edge_length1, degV1, _) = Data_Process.process_data(dataset)

Drug_Features = torch.tensor(Drug_Features).float().to(device)
Cell_Line_Feature = torch.tensor(Cell_Line_Feature).float().to(device)
V1 = torch.tensor(V1).long().to(device)
E1 = torch.tensor(E1).long().to(device)
for i in range(len(degV1)): degV1[i] = torch.tensor(degV1[i]).float().to(device)

fold0 = realFolds[0]
fold0 = Data_Process.torch_from_numpy(fold0, device)
numDrug = int(fold0.numDrug)
numCell = Cell_Line_Feature.shape[0]
numProtein = numNode1 - numDrug

cell_degV = {i: fold0.degV_dict[i].float().to(device) for i in range(len(fold0.degV_dict))}

model = Synergy_Models.Synergy(
    numDrug,
    Synergy_Models.BioEncoder(Drug_Features.shape[1], Cell_Line_Feature.shape[1], numCell, numProtein, 512, device),
    Synergy_Models.RHGNN(V1, E1, edge_num1, edge_length1, degV1, 512, 256, 256, num_edge_types=6, dropout=0),
    Synergy_Models.RHGNN(fold0.V, fold0.E, fold0.hypergraph_edge_num, fold0.edge_length, cell_degV, 512, 128, 256, num_edge_types=2, dropout=0),
    Synergy_Models.BilinearDecoder(feature_dim=256, numDrug=numDrug, cellscount=numCell, cell_dim=512,
        gene_dim=Cell_Line_Feature.shape[1], use_cell_gate=True, use_gene_synergy=True)
).to(device)

state = torch.load(ckpt, map_location=device)
filtered = {k: v for k, v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
model.load_state_dict(filtered, strict=False)
for k, v in model.state_dict().items():
    if k not in filtered: v.zero_()
model.eval()
print(f"Loaded {len(filtered)}/{len(state)} layers")

# 获取药物名和细胞系名
drug_names = dict(zip(DrugToID['Drug_ID'], DrugToID['Name'])) if 'Name' in DrugToID.columns else {}
cell_info = pd.read_csv(f"{Config.parse().data_file}/{dataset}/{dataset}_SCORE.csv", encoding='UTF-8')
cell_names = dict(enumerate(cell_info['Cell_Line'].unique()))

# 每个细胞系预测 TOP-K
TOP_K = 5
results = []
with torch.no_grad():
    for cell_idx, cell_id in enumerate(range(numDrug, numDrug + numCell)):
        # 所有药物对
        all_pairs = [(dA, dB) for dA in range(numDrug) for dB in range(dA+1, numDrug)]
        all_A = torch.tensor([p[0] for p in all_pairs], device=device)
        all_B = torch.tensor([p[1] for p in all_pairs], device=device)
        full_idx = {cid: torch.stack([all_A, all_B], dim=1) if cid == cell_id else torch.empty((0,2), dtype=torch.long, device=device)
                     for cid in range(numDrug, numDrug + numCell)}
        preds, _ = model(Drug_Features, Cell_Line_Feature, full_idx)
        scores = torch.sigmoid(preds[cell_id]).cpu().numpy()

        top_idx = np.argsort(-scores)[:TOP_K]
        for rank, idx in enumerate(top_idx):
            dA, dB = all_pairs[idx]
            name_a = drug_names.get(dA, f'Drug{dA}')
            name_b = drug_names.get(dB, f'Drug{dB}')
            cell_name = cell_names.get(cell_idx, f'Cell{cell_idx}')
            results.append({'Cell Line': cell_name, 'Drug A': name_a, 'Drug B': name_b, 'Score': f'{scores[idx]:.4f}'})

df = pd.DataFrame(results)
out = f"analysis/top_synergy_{dataset}.csv"
df.to_csv(out, index=False)
print(f"Saved to {out}")
print(df.head(20))
