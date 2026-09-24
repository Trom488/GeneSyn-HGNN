"""O'Neil 数据集灵敏度分析：不同阈值 × 超边组合 → ACC + AUPR 双热力图。"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

THRESHOLDS = [10, 15, 20, 25, 30]

EDGE_LABELS = [
    'PPI core (3)',
    'PPI+nonDDI',
    'w/o t1_3,t3 (4)',
    'All 6 types',
    'w/o t1_3 (5)',
]

# 填入你的 Mean ACC 和 Mean AUPR
data_acc = {
    # 顺序: PPI core(3), PPI+nonDDI, w/o t1_3,t3(4), All 6, w/o t1_3(5)
    30: [0.9707, 0.9697, 0.9718, 0.9702, 0.9714],
    25: [ 0.0,   0.0,   0.0,   0.0,   0.0],
    20: [ 0.0,   0.0,   0.0,   0.0,   0.0],
    15: [ 0.0,   0.0,   0.0,   0.0,   0.0],
    10: [ 0.0,   0.0,   0.0,   0.0,   0.0],
}

data_aupr = {
    30: [ 0.0,   0.0,   0.0,   0.0,   0.0],  # 填入你的 Mean AUPR
    25: [ 0.0,   0.0,   0.0,   0.0,   0.0],
    20: [ 0.0,   0.0,   0.0,   0.0,   0.0],
    15: [ 0.0,   0.0,   0.0,   0.0,   0.0],
    10: [ 0.0,   0.0,   0.0,   0.0,   0.0],
}

def make_heatmap(ax, data, title, vmin, vmax, fmt='.3f'):
    matrix = np.array([data[t] for t in THRESHOLDS])
    mask = (matrix == 0.0)
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', ['#f7fbff', '#2171b5', '#08519c'])
    im = ax.imshow(np.ma.array(matrix, mask=mask), cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(EDGE_LABELS)))
    ax.set_xticklabels(EDGE_LABELS, rotation=20, ha='right', fontsize=10)
    ax.set_yticks(range(len(THRESHOLDS)))
    ax.set_yticklabels([f'Th={t}' for t in THRESHOLDS], fontsize=10)
    for i in range(len(THRESHOLDS)):
        for j in range(len(EDGE_LABELS)):
            val = matrix[i, j]
            if val == 0.0:
                ax.text(j, i, '?', ha='center', va='center', fontsize=12, color='gray')
            else:
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center', fontsize=9,
                        color='white' if val < (vmin+vmax)/2 else 'yellow')
    ax.set_title(title, fontsize=12)
    return im

vmin = 0.88
vmax = 0.98

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
im1 = make_heatmap(ax1, data_acc, "ACC", vmin, vmax)
im2 = make_heatmap(ax2, data_aupr, "AUPR", vmin, vmax, fmt='.4f')

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)
fig.suptitle("Sensitivity Analysis: O'Neil Dataset", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("analysis/sensitivity_heatmap.png", dpi=200, bbox_inches='tight')
print("Saved: analysis/sensitivity_heatmap.png")

# 待跑命令
missing = False
for i in range(len(THRESHOLDS)):
    for j in range(len(EDGE_LABELS)):
        if data_acc[THRESHOLDS[i]][j] == 0.0:
            if not missing:
                print("\n待跑实验:")
                missing = True
            edge_map = {0: '0,2,5', 1: '0,1,5', 2: '0,2,4,5', 3: '0,1,2,3,4,5', 4: '0,2,3,4,5'}
            print(f"  python GeneSyn-HGNN.py --dataset ONEIL --epochs 1500 --cuda 0 --threshold {THRESHOLDS[i]} --edge_types {edge_map[j]}")
