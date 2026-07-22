import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# BlockTxCircuit data (sorted by cost, no VastAI)
btx_labels = [
    'g6e spot\nL40S',
    'mac-m4\nMetal M4',
    'mac-m4\nCPU M4',
    'g6e on-dem\nL40S',
    'g6e spot\nCPU EPYC',
]
btx_costs = [0.000746, 0.000858, 0.001394, 0.001494, 0.001629]
btx_colors = ['#3498db', '#e67e22', '#bdc3c7', '#3498db', '#bdc3c7']

# BlockTxChainCircuit data (sorted by cost, no VastAI)
btc_labels = [
    'mac-m4\nMetal M4',
    'g6e spot\nL40S',
    'mac-m4\nCPU M4',
    'g6e on-dem\nL40S',
    'g6e spot\nCPU EPYC',
]
btc_costs = [0.000178, 0.000196, 0.000249, 0.000391, 0.000439]
btc_colors = ['#e67e22', '#3498db', '#bdc3c7', '#3498db', '#bdc3c7']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# --- BlockTxCircuit ---
bars1 = ax1.bar(range(len(btx_labels)), btx_costs, color=btx_colors, edgecolor='#2c3e50', linewidth=0.8, width=0.6)
ax1.set_xticks(range(len(btx_labels)))
ax1.set_xticklabels(btx_labels, fontsize=9)
ax1.set_ylabel('Cost per Proof (USD)', fontsize=11)
ax1.set_title('BlockTxCircuit (Large, degree 2\u00b9\u2076)\n85% of total proving time', fontsize=12, fontweight='bold')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y*1e6:.0f}\u00d710\u207b\u2076' if y > 0 else '$0'))

for bar, cost in zip(bars1, btx_costs):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(btx_costs) * 0.02,
             f'${cost*1e6:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylim(0, max(btx_costs) * 1.18)
ax1.grid(axis='y', alpha=0.3)

# --- BlockTxChainCircuit ---
bars2 = ax2.bar(range(len(btc_labels)), btc_costs, color=btc_colors, edgecolor='#2c3e50', linewidth=0.8, width=0.6)
ax2.set_xticks(range(len(btc_labels)))
ax2.set_xticklabels(btc_labels, fontsize=9)
ax2.set_ylabel('Cost per Proof (USD)', fontsize=11)
ax2.set_title('BlockTxChainCircuit (Small, degree 2\u00b9\u2074)', fontsize=12, fontweight='bold')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y*1e6:.0f}\u00d710\u207b\u2076' if y > 0 else '$0'))

for bar, cost in zip(bars2, btc_costs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(btc_costs) * 0.02,
             f'${cost*1e6:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylim(0, max(btc_costs) * 1.18)
ax2.grid(axis='y', alpha=0.3)

# Legend
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='#2c3e50', label='AWS CUDA (g6e, L40S)'),
    Patch(facecolor='#e67e22', edgecolor='#2c3e50', label='AWS Metal (mac-m4, M4)'),
    Patch(facecolor='#bdc3c7', edgecolor='#2c3e50', label='CPU-only (no GPU)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle('Plonky2 Lighter-Prover: Cost per Proof on AWS (\u00d710\u207b\u2076 USD)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/fujie/coding/cysic/20260220/elliottech-plonky2/cost_per_proof.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved to cost_per_proof.png")
