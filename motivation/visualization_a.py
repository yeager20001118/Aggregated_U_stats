import matplotlib.pyplot as plt
import numpy as np

n = [100, 200, 300, 400, 500]

original = np.array([20.0, 52.0, 86.0, 96.0, 100.0]) / 100 # 20 kernels
r1 = np.array([24.0, 67.0, 93.0, 99.0, 100.0]) / 100 # seed 16
r2 = np.array([23.0, 60.0, 88.0, 96.0, 100.0]) / 100 # seed 91
r3 = np.array([12.0, 24.0, 35.0, 38.0, 92.0]) / 100 # seed 31
r4 = np.array([7.0, 14.0, 18.0, 27.0, 68.0]) / 100 # seed 1
r5 = np.array([4.0, 3.0, 4.0, 4.0, 9.0]) / 100 # seed 26

plt.figure(figsize=(6.5, 5.5))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set proper limits first
plt.xlim([75, 530])  # Extended for arrow
plt.ylim([-0.05, 1.10])  # Extended for arrow

# Position the axes correctly (only once)
ax.spines['left'].set_position(('data', 80))
ax.spines['bottom'].set_position(('data', -0.02))

# Add proper arrows to axes with appropriate sizes
# X-axis arrow
plt.arrow(520.15, -0.0213, 10, 0, shape='full', lw=0,
          length_includes_head=True, head_width=0.025, head_length=14, fc='black')
# Y-axis arrow
plt.arrow(80, 1.05, 0, 0.05, shape='full', lw=0,
          length_includes_head=True, head_width=10, head_length=0.04, fc='black')

plt.xlabel('Sample Size', fontsize=22)
plt.ylabel('Test Power', fontsize=22)

# X[-1] = 0.35
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
plt.xticks([100, 300, 500], fontsize=18)
# plt.xlabel('Number of Samples', fontsize=22)
# plt.ylabel('$(a)$ Kernel Density Estimation', fontsize=22)

plt.plot(n, original, ls='-', linewidth=2, marker='o', label='set of 20 kernels')
plt.plot(n, r1, ls='--', linewidth=2, marker='s', label='$subset1$')
plt.plot(n, r2, ls='--', linewidth=2, marker='^', label='$subset2$')
plt.plot(n, r3, ls='--', linewidth=2, marker='d', label='$subset3$')
plt.plot(n, r4, ls='--', linewidth=2, marker='x', label='$subset4$')
plt.plot(n, r5, ls='--', linewidth=2, marker='+', label='$subset5$')

plt.legend(bbox_to_anchor=[0.02, 1.0],
           loc='upper left', fontsize=13, frameon=False)


plt.tight_layout()
plt.savefig('Subset_Kernel.pdf', bbox_inches='tight')
plt.show()
