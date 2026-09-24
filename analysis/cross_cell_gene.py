"""跨细胞系基因归因对比：同一药物对在不同细胞系的关键基因差异。"""
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
 _, _, CellLineToID, V1, E1, edge_num1, edge_length1, degV1, _) = Data_Process.process_data(dataset)

Drug_Features = torch.tensor(Drug_Features).float().to(device)
Cell_Line_Feature_raw = torch.tensor(Cell_Line_Feature).float().to(device)
V1 = torch.tensor(V1).long().to(device)
E1 = torch.tensor(E1).long().to(device)
for i in range(len(degV1)): degV1[i] = torch.tensor(degV1[i]).float().to(device)

fold0 = realFolds[0]
fold0 = Data_Process.torch_from_numpy(fold0, device)
numDrug = int(fold0.numDrug)
numCell = Cell_Line_Feature.shape[0]
cell_degV = {i: fold0.degV_dict[i].float().to(device) for i in range(len(fold0.degV_dict))}

model = Synergy_Models.Synergy(
    numDrug,
    Synergy_Models.BioEncoder(Drug_Features.shape[1], Cell_Line_Feature.shape[1], numCell, numNode1-numDrug, 512, device),
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

# 基因名
path = f"{Config.parse().data_file}/{dataset}/{dataset}_CELL_LINE_EXPRESSION.csv"
gene_df = pd.read_csv(path, nrows=0)
gene_cols = list(gene_df.columns[1:])
cell_names = list(CellLineToID['Cell_Line']) if 'Cell_Line' in CellLineToID.columns else [f'C{i}' for i in range(numCell)]

# ====== 找跨细胞系预测差异最大的药物对 ======
print("Searching for drug pair with maximal cell-line divergence...")
best_diff, best_pair = 0, (0, 0)
with torch.no_grad():
    for dA in range(min(8, numDrug)):
        for dB in range(dA+1, min(8, numDrug)):
            full_idx = {cid: torch.tensor([[dA, dB]], device=device) if cid - numDrug < 10 else torch.empty((0,2), dtype=torch.long, device=device)
                         for cid in range(numDrug, numDrug + numCell)}
            preds, _ = model(Drug_Features, Cell_Line_Feature_raw, full_idx)
            scores = {cid: torch.sigmoid(preds[cid]).item() for cid in range(numDrug, min(numDrug+10, numDrug+numCell))}
            diff = max(scores.values()) - min(scores.values())
            if diff > best_diff:
                best_diff, best_pair = diff, (dA, dB)
                best_scores = scores

dA, dB = best_pair
# 找最高和最低的细胞系
high_cell = max(best_scores, key=best_scores.get)
low_cell = min(best_scores, key=best_scores.get)
print(f"Drug pair: {dA},{dB} | High: cell {high_cell} ({best_scores[high_cell]:.4f}) | Low: cell {low_cell} ({best_scores[low_cell]:.4f}) | Δ={best_diff:.4f}")

# ====== 基因梯度对比 ======
def gene_importance(cell_line_id):
    Cell_Line_Feature_raw.grad = None
    Cell_Line_Feature_raw.requires_grad_(True)
    x_drug, x_cell, x_protein = model.BioEncoder(Drug_Features, Cell_Line_Feature_raw)
    emb1 = model.hgnn_encoder1(torch.cat([x_drug, x_protein], 0))
    emb2 = model.hgnn_encoder2(torch.cat([x_drug, x_cell], 0))
    emb = (emb1[:numDrug] + emb2[:numDrug]) / 2
    DrugA, DrugB = emb[[dA]], emb[[dB]]
    cell_vec = x_cell[cell_line_id - numDrug]
    score = (DrugA * DrugB * (cell_vec @ model.decoder.W_gs)).sum()
    score.backward()
    return Cell_Line_Feature_raw.grad[cell_line_id - numDrug].abs().cpu().numpy()

high_grad = gene_importance(high_cell)
low_grad = gene_importance(low_cell)

# 输出
h_name = cell_names[high_cell - numDrug] if high_cell - numDrug < len(cell_names) else f'Cell{high_cell}'
l_name = cell_names[low_cell - numDrug] if low_cell - numDrug < len(cell_names) else f'Cell{low_cell}'

print(f"\n=== 跨细胞系基因归因: drug ({dA},{dB}) ===")
print(f"{'Rank':<6} {h_name:<25} {'Imp':>10} | {l_name:<25} {'Imp':>10}")
print("-" * 80)
for rank in range(20):
    h_idx = int(np.argsort(-high_grad)[rank])
    l_idx = int(np.argsort(-low_grad)[rank])
    h_gene = gene_cols[h_idx] if h_idx < len(gene_cols) else f'G{h_idx}'
    l_gene = gene_cols[l_idx] if l_idx < len(gene_cols) else f'G{l_idx}'
    print(f"{rank+1:<6} {h_gene:<25} {high_grad[h_idx]:10.6f} | {l_gene:<25} {low_grad[l_idx]:10.6f}")
