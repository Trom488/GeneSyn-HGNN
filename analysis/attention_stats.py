"""注意力权重统计：每类超边的平均注意力，验证 PPI 边获得更高关注。"""
import torch, numpy as np, sys, os
sys.path.insert(0, '.')
import Config, Data_Process, Synergy_Models

args = Config.parse()
dataset = args.dataset
device = args.device

ckpt = f"checkpoints/{dataset}_pretrained.pt"
if not os.path.exists(ckpt):
    print(f"No checkpoint: {ckpt}"); sys.exit(1)

(realFolds, _, _, _, numNode1, Drug_Features, Cell_Line_Feature,
 _, _, _, V1, E1, edge_num1, edge_length1, degV1, _) = Data_Process.process_data(dataset)

Drug_Features = torch.tensor(Drug_Features).float().to(device)
Cell_Line_Feature = torch.tensor(Cell_Line_Feature).float().to(device)
V1 = torch.tensor(V1).long().to(device)
E1 = torch.tensor(E1).long().to(device)
for i in range(len(degV1)): degV1[i] = torch.tensor(degV1[i]).float().to(device)

fold0 = realFolds[0]
fold0 = Data_Process.torch_from_numpy(fold0, device)
numDrug = int(fold0.numDrug)

model = Synergy_Models.Synergy(
    numDrug,
    Synergy_Models.BioEncoder(Drug_Features.shape[1], Cell_Line_Feature.shape[1], Cell_Line_Feature.shape[0], numNode1-numDrug, 512, device),
    Synergy_Models.RHGNN(V1, E1, edge_num1, edge_length1, degV1, 512, 256, 256, num_edge_types=6, dropout=0),
    Synergy_Models.RHGNN(torch.zeros(0), torch.zeros(0), 0, [0], {}, 512, 128, 256, num_edge_types=2, dropout=0),
    Synergy_Models.BilinearDecoder(feature_dim=256, numDrug=numDrug, cellscount=Cell_Line_Feature.shape[0], cell_dim=512)
).to(device)

state = torch.load(ckpt, map_location=device)
filtered = {k: v for k, v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
model.load_state_dict(filtered, strict=False)
model.eval()

# 提取 conv_in 的注意力权重
conv = model.hgnn_encoder1.conv_in
type_names = ['t1(PPI+)', 't1_3(triangle)', 't2(PPI-)', 't3(DDI+PPI)', 't4(DDI-PPI)', 't5(nonDDI)']

with torch.no_grad():
    x_drug, x_cell, x_protein = model.BioEncoder(Drug_Features, Cell_Line_Feature)
    X = torch.cat([x_drug, x_protein], 0)

    attn_means = []
    for i in range(6):
        node_feat = conv.W[i+1](X) * conv.degV_dict[i]
        h = node_feat[conv.V]
        attn = conv.leaky_relu(h @ conv.a_n2e[i] / (conv.out_channels ** 0.5))
        alpha = Synergy_Models.safe_scatter_softmax(attn, conv.E, dim_size=conv.edge_num)

        # 该类型的超边范围
        start = 0 if i == 0 else conv.Xe_class_length[i-1]
        end = conv.Xe_class_length[i]
        if end > start:
            type_alpha = alpha[start:end]  # 仅该类型超边的注意力
            attn_means.append((type_names[i], float(type_alpha.mean()), float(type_alpha.std()), end-start))
        else:
            attn_means.append((type_names[i], 0.0, 0.0, 0))

print(f"\n=== 节点→超边 注意力统计 ({dataset}) ===")
print(f"{'超边类型':<18} {'数量':>8} {'均值':>10} {'标准差':>8}")
print("-" * 50)
for name, mean_val, std_val, count in attn_means:
    print(f"{name:<18} {count:>8} {mean_val:>10.6f} {std_val:>8.6f}")
