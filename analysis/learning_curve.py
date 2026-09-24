"""学习曲线：命题 4（样本复杂度）的实证检验。

比较 默认模型（带 gene_synergy） 与 w/o gene_synergy 在不同训练比例下的测试 AUC/AUPR，
检验"结构化低秩项达到目标精度所需样本更少（学习曲线整体左移）"的预言。
注意：正确读数是水平读数（达到同一 AUC 所需训练比例），而非同比例下的垂直差值——
低样本区两模型均欠拟合，垂直差距被压缩，优势以样本效率形式体现。

用法:
  python analysis/learning_curve.py --dataset ONEIL --epochs 800
  python analysis/learning_curve.py --epochs 300 --fractions 0.2,0.4,0.6   # 快速版

说明:
  - 训练比例通过 train_fraction 在每个细胞系内按比例保留训练样本，test/valid 不受影响，
    因此各 fraction 共享同一测试集。
  - 两种配置同种子、同折结构、同 epochs 下对比，只有 gene_synergy 开关不同。
  - 相对比较在相同 epochs 下才公平，建议扫描用较短 epochs（如 500-800）。
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

# GeneSyn-HGNN.py 文件名含连字符，无法直接 import，用 importlib 加载以复用 train/test/metrics
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gensyn", os.path.join(_here, "..", "GeneSyn-HGNN.py"))
gensyn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gensyn)


def evaluate_config(args, tensors, folds, drug_target_count):
    """按当前 args（train_fraction + no_gene_synergy）跑 k 折，返回 (AUC, AUPR) 的 (mean, std)。"""
    Drug_Features, Cell_Line_Feature, V1, E1, edge_num1, edge_length1, degV1, numNode1 = tensors
    aucs, auprs = [], []
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
                                 Data.degV_dict, 512, 128, 256, num_edge_types=2, dropout=0),
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
        gensyn.DrugCellIndex = {}        # 仅 beta>0 时用到，默认 0 则安全
        gensyn.RelationNet = None

        with mlflow.start_run():
            mlflow.log_param("train_fraction", args.train_fraction)
            mlflow.log_param("no_gene_synergy", bool(args.no_gene_synergy))
            best_state, _ = gensyn.train(Drug_Features, Data, args.epochs, scheduler, drug_target_tensor)
            Model.load_state_dict(best_state)
            _, test_pred, test_real = gensyn.test(Data.test_edges, Data.test_labels, drug_target_tensor)
            auc, aupr, _, _ = gensyn.metrics(test_real, test_pred, 0, 'learning_curve_test')

        aucs.append(auc)
        auprs.append(aupr)

    return (float(np.mean(aucs)), float(np.std(aucs))), (float(np.mean(auprs)), float(np.std(auprs)))


def main():
    args = Config.parse()
    fractions = args.fractions
    configs = [("full", False), ("w/o gene_synergy", True)]

    print(f"Learning curve: dataset={args.dataset}, epochs={args.epochs}, "
          f"fractions={fractions}, k_fold={args.k_fold}, split={args.split_mode}")
    print("Configs: " + ", ".join(c for c, _ in configs))
    print("-" * 78)

    rows = []
    for frac in fractions:
        for cfg_name, no_gs in configs:
            args.train_fraction = frac
            args.no_gene_synergy = no_gs
            # produce() 读的是 Data_Process 模块级 params，需同步覆盖
            Data_Process.params.train_fraction = frac
            Data_Process.params.split_mode = args.split_mode

            # 每个 fraction 重建 folds（训练下采样发生在 produce 内）
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

            (m_auc, s_auc), (m_aupr, s_aupr) = evaluate_config(args, tensors, folds, drug_target_count)
            rows.append((frac, cfg_name, m_auc, s_auc, m_aupr, s_aupr))
            print(f"frac={frac:.2f}  {cfg_name:<16}  AUC={m_auc:.4f}+/-{s_auc:.4f}  "
                  f"AUPR={m_aupr:.4f}+/-{s_aupr:.4f}", flush=True)

    print("\n" + "=" * 78)
    print("Summary (AUC):  delta = full - w/o gene_synergy（>0 表示 gene_synergy 有贡献）")
    print(f"{'fraction':>10} | {'full':>14} | {'w/o gene_synergy':>17} | {'delta':>10}")
    for frac in fractions:
        full = next(r for r in rows if r[0] == frac and r[1] == "full")
        no_gs = next(r for r in rows if r[0] == frac and r[1] == "w/o gene_synergy")
        print(f"{frac:10.2f} | {full[2]:.4f}+/-{full[3]:.4f} | {no_gs[2]:.4f}+/-{no_gs[3]:.4f} | {full[2]-no_gs[2]:+.4f}")

    _try_plot(args, fractions, rows)


def _try_plot(args, fractions, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[skip plot] matplotlib 不可用: {e}")
        return

    plt.figure(figsize=(6, 4.5))
    for cfg_name in ["full", "w/o gene_synergy"]:
        xs = [r[0] for r in rows if r[1] == cfg_name]
        ys = [r[2] for r in rows if r[1] == cfg_name]
        std = [r[3] for r in rows if r[1] == cfg_name]
        label = cfg_name
        plt.errorbar(xs, ys, yerr=std, marker="o", label=label, capsize=3)
    plt.xlabel("train fraction")
    plt.ylabel("test AUC")
    plt.title(f"Learning curve ({args.dataset})")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(_here, f"learning_curve_{args.dataset}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
