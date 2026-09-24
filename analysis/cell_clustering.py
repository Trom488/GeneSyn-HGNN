"""W_gs 细胞系聚类：提取细胞系在基因-协同空间的表示，验证生物学一致性。"""
import torch, numpy as np, pandas as pd, sys, os
sys.path.insert(0, '.')
import Config, Data_Process, Synergy_Models
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

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

# 提取每个细胞系的 W_gs 变换表示
with torch.no_grad():
    x_drug, x_cell, x_protein = model.BioEncoder(Drug_Features, Cell_Line_Feature)
    W_gs = model.decoder.W_gs  # [512, 256]
    gs_repr = torch.matmul(x_cell, W_gs).cpu().numpy()  # [numCell, 256]

# 细胞系名
cell_names = list(CellLineToID['Cell_Line']) if 'Cell_Line' in CellLineToID.columns else [f'Cell{i}' for i in range(numCell)]

# ========== 层次聚类（自动选 k） ==========
Z = linkage(gs_repr, method='ward')
best_k = min(5, numCell)
labels = fcluster(Z, best_k, criterion='maxclust')
sil = silhouette_score(gs_repr, labels)

print(f"\n=== W_gs 细胞系聚类 (k={best_k}, silhouette={sil:.3f}) ===")
for c in range(1, best_k+1):
    members = [cell_names[i] for i in range(numCell) if labels[i] == c]
    print(f"  Cluster {c} ({len(members)} cells): {', '.join(members)}")

# ========== t-SNE 可视化 ==========
tsne = TSNE(n_components=2, perplexity=min(15, numCell-1), random_state=42)
gs_2d = tsne.fit_transform(gs_repr)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for c in range(1, best_k+1):
    mask = labels == c
    ax.scatter(gs_2d[mask, 0], gs_2d[mask, 1], c=[colors[c-1]], label=f'Cluster {c}', s=80)
for i, name in enumerate(cell_names):
    ax.annotate(name[:12], (gs_2d[i, 0], gs_2d[i, 1]), fontsize=7, alpha=0.7)
ax.set_title("Cell Line Clustering via W_gs Embeddings\n(Gene-Synergy Space)", fontsize=13)
ax.legend()
plt.tight_layout()
out_png = f"analysis/cell_clustering_{dataset}.png"
plt.savefig(out_png, dpi=150)
print(f"Saved: {out_png}")
