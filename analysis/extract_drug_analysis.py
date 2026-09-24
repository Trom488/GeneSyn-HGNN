"""分析脚本 1：提取药物 embedding，层次聚类，与 ATC 对照。"""
import torch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict
import urllib.request
import json
import sys
sys.path.insert(0, '.')
import Config
import Data_Process
import Synergy_Models

args = Config.parse()

def get_atc_codes(pubchem_cids):
    """从 PubChem PUG REST API 获取 ATC 编码。"""
    atc_map = {}
    batch_size = 20
    for i in range(0, len(pubchem_cids), batch_size):
        batch = pubchem_cids[i:i+batch_size]
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str,batch))}/property/ATC_Code/JSON"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
                for prop in data.get('PropertyTable', {}).get('Properties', []):
                    cid = prop['CID']
                    atc = prop.get('ATC_Code', '')
                    if atc:
                        atc_map[cid] = atc.split('/')[0]  # 只取一级 ATC 分类
        except Exception as e:
            print(f"  Warning: fetch ATC for batch {i} failed: {e}")
    return atc_map


def extract_drug_embeddings(checkpoint_path, dataset='ONEIL'):
    """从保存的模型 checkpoint 提取药物 embedding。"""
    print(f"Loading checkpoint: {checkpoint_path}")
    _, Protein_Adjacency_Matrix, _, _, numNode1, Drug_Features, Cell_Line_Feature, DrugToID, _, _, V1, E1, edge_num1, edge_length1, degV1, drug_target_count = Data_Process.process_data(dataset)

    device = args.device
    Drug_Features = torch.tensor(Drug_Features).float().to(device)
    Cell_Line_Feature = torch.tensor(Cell_Line_Feature).float().to(device)
    V1 = torch.tensor(V1).long().to(device)
    E1 = torch.tensor(E1).long().to(device)
    for i in range(len(degV1)):
        degV1[i] = torch.tensor(degV1[i]).float().to(device)

    bio_encoder = Synergy_Models.BioEncoder(Drug_Features.shape[1], Cell_Line_Feature.shape[1], 39,
                                              numNode1 - 38, 512, device)
    hgnn1 = Synergy_Models.RHGNN(V1, E1, edge_num1, edge_length1, degV1, 512, 256, 256, num_edge_types=6, dropout=0)
    hgnn2 = Synergy_Models.RHGNN(torch.zeros(0), torch.zeros(0), 0, [0], {}, 512, 256, 256, num_edge_types=2, dropout=0)
    decoder = Synergy_Models.BilinearDecoder(feature_dim=256, numDrug=38, cellscount=39, cell_dim=512)

    model = Synergy_Models.Synergy(38, bio_encoder, hgnn1, hgnn2, decoder).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        x_drug, x_cell, x_protein = model.BioEncoder(Drug_Features, Cell_Line_Feature)
        hg1 = torch.cat([x_drug, x_protein], 0)
        emb1 = model.hgnn_encoder1(hg1)
        drug_emb = emb1[:38, :]  # Channel 1 药物表示
        # 也取融合 embedding
        hg2 = torch.cat([x_drug, x_cell], 0)
        emb2 = model.hgnn_encoder2(hg2)
        drug_fused = (emb1[:38, :] + emb2[:38, :]) / 2

    return drug_emb.cpu().numpy(), drug_fused.cpu().numpy(), DrugToID


def cluster_evaluate(embeddings, labels, n_clusters):
    """层次聚类并评估与真实标签的一致性。"""
    dist = pdist(embeddings, metric='cosine')
    Z = linkage(dist, method='ward')
    pred = fcluster(Z, n_clusters, criterion='maxclust')
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    return ari, nmi, pred


if __name__ == '__main__':
    import os
    ckpt = f"checkpoints/{args.dataset}_pretrained.pt"
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        print("Run training first to generate checkpoint.")
        sys.exit(1)

    drug_emb, drug_fused, DrugToID = extract_drug_embeddings(ckpt, args.dataset)
    print(f"Extracted drug embeddings: {drug_emb.shape}")

    # 获取 ATC 编码
    pubchem_cids = DrugToID['PubChem_CID'].tolist()
    print(f"Fetching ATC codes for {len(pubchem_cids)} drugs...")
    atc_map = get_atc_codes(pubchem_cids)
    print(f"  Got ATC for {len(atc_map)}/{len(pubchem_cids)} drugs")

    # 构建 ATC 标签
    atc_labels = []
    valid_indices = []
    for i, cid in enumerate(pubchem_cids):
        if cid in atc_map:
            atc_labels.append(atc_map[cid])
            valid_indices.append(i)

    if len(set(atc_labels)) < 2:
        print("Not enough ATC categories for clustering evaluation.")
        # Fallback: 用药物靶点相似度分组
        print("Using target-based pseudo-labels instead...")

    else:
        # 聚类评估
        emb_subset = drug_fused[valid_indices]
        n_clusters = min(len(set(atc_labels)), 8)
        ari, nmi, pred = cluster_evaluate(emb_subset, atc_labels, n_clusters)
        print(f"\n=== 聚类 vs ATC ===")
        print(f"  ARI (Adjusted Rand Index): {ari:.4f}  (1.0=完美, 0=随机)")
        print(f"  NMI (Normalized Mutual Info): {nmi:.4f}")
        print(f"  Clusters: {n_clusters}, unique ATC: {len(set(atc_labels))}")

        # 输出每个簇的药物
        clusters = defaultdict(list)
        for i, (idx, label) in enumerate(zip(valid_indices, pred)):
            drug_name = DrugToID.iloc[idx]['Name'] if 'Name' in DrugToID.columns else f"Drug{idx}"
            clusters[int(label)].append(f"{drug_name}({atc_labels[i]})")
        print(f"\n=== 药物簇 ===")
        for cid, drugs in sorted(clusters.items()):
            print(f"  Cluster {cid}: {', '.join(drugs[:5])}{'...' if len(drugs)>5 else ''}")
