"""RHGNN2 dropout 消融：细胞-药物协同/拮抗超图编码器的 dropout 设置。

RHGNN1（药物-蛋白超图，固定生物结构）dropout=0.2 固定；仅切换 RHGNN2 的 dropout。
RHGNN2 的超边直接来自训练标签（协同/拮抗对），是全模型过拟合风险最高的模块，
本实验检验其 dropout=0 是否合理。

用法:
  python analysis/rhgnn2_dropout.py --dataset ONEIL --epochs 800
  python analysis/rhgnn2_dropout.py --dropouts 0,0.1,0.2 --epochs 300

说明:
  - 各配置同种子、同折结构、同 epochs，仅 RHGNN2 dropout 不同，结果可直接比较。
  - 评估在共享测试集上，报告 AUC/AUPR/F1/ACC 的 mean±std（k 折）。
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util

import numpy as np
import torch
import mlflow

import Config
import Data_Process
import Synergy_Models

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gensyn", os.path.join(_here, "..", "GeneSyn-HGNN.py"))
gensyn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gensyn)


def evaluate_dropout2(args, tensors, folds, drug_target_count, dropout2):
    Drug_Features, Cell_Line_Feature, V1, E1, edge_num1, edge_length1, degV1, numNode1 = tensors
    fold_metrics = []
    for fold_data in folds:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        Data = Data_Process.torch_from_numpy(fold_data, args.device)

        drug_target_tensor = torch.zeros(Data.numDrug, device=args.device)
        for drug_id, count in drug_target_count.items():
            if drug_id < Data.numDrug:
                drug_target_tensor[drug_id] = float(count)

        Model = Synergy_Models.Synergy(
            Data.numDrug,
            Synergy_Models.BioEncoder(Drug_Features.shape[1], Cell_Line_Feature.shape[1],
                                      Data.CellsCount, numNode1 - Data.numDrug, 512, device=args.device),
            Synergy_Models.RHGNN(V1, E1, edge_num1, edge_length1, degV1, 512, 256, 256,
                                 num_edge_types=6, dropout=0.2),
            Synergy_Models.RHGNN(Data.V, Data.E, Data.hypergraph_edge_num, Data.edge_length,
                                 Data.degV_dict, 512, 128, 256, num_edge_types=2, dropout=dropout2),
            Synergy_Models.BilinearDecoder(feature_dim=256, numDrug=Data.numDrug, cellscount=Data.CellsCount,
                                           cell_dim=512, gene_dim=Cell_Line_Feature.shape[1],
                                           use_cell_gate=not args.no_cell_gate,
                                           use_gene_synergy=not args.no_gene_synergy,
                                           use_gene_attn=not args.no_gene_attn),
        ).to(args.device)

        optimizer = torch.optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        # GeneSyn-HGNN.py 的 train/test 引用模块级全局变量，逐一注入
        gensyn.args = args
        gensyn.Model = Model
        gensyn.Drug_Features = Drug_Features
        gensyn.Cell_Line_Feature = Cell_Line_Feature
        gensyn.Data = Data
        gensyn.optimizer = optimizer
        gensyn.DrugCellIndex = {}
        gensyn.RelationNet = None

        with mlflow.start_run():
            mlflow.log_param("rhgnn2_dropout", dropout2)
            best_state, _ = gensyn.train(Drug_Features, Data, args.epochs, scheduler, drug_target_tensor)
            Model.load_state_dict(best_state)
            _, test_pred, test_real = gensyn.test(Data.test_edges, Data.test_labels, drug_target_tensor)
            auc, aupr, f1, acc = gensyn.metrics(test_real, test_pred, 0, 'rhgnn2_dropout_test')

        fold_metrics.append((auc, aupr, f1, acc))

    arr = np.array(fold_metrics)
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    args = Config.parse()
    dropouts = args.dropouts

    print(f"RHGNN2 dropout ablation: dataset={args.dataset}, epochs={args.epochs}, "
          f"k_fold={args.k_fold}, split={args.split_mode}, dropouts={dropouts}")
    print("-" * 78)

    # 各配置共享同一份折结构与数据，仅 dropout 不同
    Data_Process.params.train_fraction = 1.0
    Data_Process.params.split_mode = args.split_mode
    (realFolds, _, _, _, numNode1, Drug_Features, Cell_Line_Feature, _, _, _, V1, E1,
     edge_num1, edge_length1, degV1, drug_target_count) = Data_Process.process_data(args.dataset)

    Drug_Features = torch.tensor(Drug_Features).float().to(args.device)
    Cell_Line_Feature = torch.tensor(Cell_Line_Feature).float().to(args.device)
    V1 = torch.tensor(V1).long().to(args.device)
    E1 = torch.tensor(E1).long().to(args.device)
    for i in range(len(degV1)):
        degV1[i] = torch.tensor(degV1[i]).float().to(args.device)
    tensors = (Drug_Features, Cell_Line_Feature, V1, E1, edge_num1, edge_length1, degV1, numNode1)
    folds = [realFolds[f] for f in range(args.k_fold)]

    print(f"{'dropout2':>9} | {'AUC':>14} | {'AUPR':>14} | {'F1':>14} | {'ACC':>14}")
    rows = []
    for d in dropouts:
        (m_auc, m_aupr, m_f1, m_acc), (s_auc, s_aupr, s_f1, s_acc) = evaluate_dropout2(
            args, tensors, folds, drug_target_count, d)
        rows.append((d, (m_auc, s_auc), (m_aupr, s_aupr), (m_f1, s_f1), (m_acc, s_acc)))
        print(f"{d:9.1f} | {m_auc:.4f}+/-{s_auc:.4f} | {m_aupr:.4f}+/-{s_aupr:.4f} | "
              f"{m_f1:.4f}+/-{s_f1:.4f} | {m_acc:.4f}+/-{s_acc:.4f}", flush=True)

    best = max(rows, key=lambda r: r[1][0])  # 按 AUC 取最优
    print("-" * 78)
    print(f"Best RHGNN2 dropout by AUC: {best[0]:.1f} (AUC {best[1][0]:.4f})")

    _try_plot(args, rows)


def _try_plot(args, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[skip plot] matplotlib 不可用: {e}")
        return

    ds = [r[0] for r in rows]
    auc_mean = [r[1][0] for r in rows]
    auc_std = [r[1][1] for r in rows]
    plt.figure(figsize=(6, 4.5))
    plt.errorbar(ds, auc_mean, yerr=auc_std, marker="o", capsize=3, label="AUC")
    plt.xlabel("RHGNN2 dropout")
    plt.ylabel("test AUC")
    plt.title(f"RHGNN2 dropout ablation ({args.dataset})")
    plt.grid(alpha=0.3)
    out = os.path.join(_here, f"rhgnn2_dropout_{args.dataset}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
