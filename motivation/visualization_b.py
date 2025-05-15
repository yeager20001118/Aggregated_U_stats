import matplotlib.pyplot as plt
import numpy as np

# n = [100, 200, 300, 400, 500]
n = [50, 100, 150, 200, 250, 300, 350]
# worst = [0.02, 0.03, 0.03, 0.04, 0.08]  # 0.658
worst = [0.02, 0.02, 0.1, 0.06, 0.1, 0.08, 0.01] #.00074 laplace
best1 = [0.15, 0.24, 0.3, 0.52, 0.65, 0.71, 0.91] #.072
best2 = [0.01, 0.2, 0.14, 0.37, 0.4, 0.47, 0.46] #.009
best3 = [0.13, 0.07, 0.18, 0.21, 0.24, 0.21, 0.34] #.110

plt.figure(figsize=(9, 5))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set proper limits first
plt.xlim([25, 380])  # Extended for arrow
plt.ylim([-0.05, 1.10])  # Extended for arrow

# Position the axes correctly (only once)
ax.spines['left'].set_position(('data', 30))
ax.spines['bottom'].set_position(('data', -0.02))

# Add proper arrows to axes with appropriate sizes
# X-axis arrow
plt.arrow(370.15, -0.0213, 10, 0, shape='full', lw=0,
          length_includes_head=True, head_width=0.05, head_length=12, fc='black')
# Y-axis arrow
plt.arrow(30, 1.05, 0, 0.05, shape='full', lw=0,
          length_includes_head=True, head_width=8, head_length=0.06, fc='black')

plt.xlabel('Sample Size', fontsize=22)
plt.ylabel('Test Power', fontsize=22)

# X[-1] = 0.35
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
plt.xticks([50, 100, 150, 200, 250, 300, 350], fontsize=18)
# plt.xlabel('Number of Samples', fontsize=22)
# plt.ylabel('$(a)$ Kernel Density Estimation', fontsize=22)

plt.plot(n, np.array(best1), ls='-', linewidth=2, marker='o', label='$\kappa_1$=.072', color='blue')
plt.plot(n, best2, ls='--', linewidth=2, marker='d', label='$\kappa_2$=.009', color='orange')
plt.plot(n, best3, ls='--', linewidth=2, marker='*', label='$\kappa_3$=.110', color='green')
plt.plot(n, worst, ls='--', linewidth=2, marker='s', label='$\kappa_4$=.00074', color='red')

plt.legend(bbox_to_anchor=[0.06, 1.0],
           loc='upper left', fontsize=15, frameon=False)


plt.tight_layout()
plt.savefig('Kernel.pdf', bbox_inches='tight')
plt.show()

# n = [50, 100, 150, 200, 250, 300, 350]
# [0.15, 0.24, 0.3, 0.52, 0.65, 0.71, 0.91] #.072
# [0.01, 0.2, 0.14, 0.37, 0.4, 0.47, 0.46] #.009
# [0.13, 0.07, 0.18, 0.21, 0.24, 0.21, 0.34] #.110
# [0.06, 0.03, 0.06, 0.04, 0.06, 0.07, 0.05] #.490

# [0.1, 0.14, 0.22, 0.51, 0.64, 0.71, 0.8] #+.009
# [0.24, 0.3, 0.58, 0.82, 0.95, 0.98, 1.0] #+.110
# [0.13, 0.18, 0.4, 0.62, 0.76, 0.86, 0.93] #+.490
