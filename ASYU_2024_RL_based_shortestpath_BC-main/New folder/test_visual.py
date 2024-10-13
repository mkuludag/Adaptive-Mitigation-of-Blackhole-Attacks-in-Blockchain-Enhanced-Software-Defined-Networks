import numpy as np
import matplotlib.pyplot as plt

# Data arrays
qc = [563, 800, 800, 800, 800]
hra = [0, 0, 0, 483, 595]
dra = [200, 203, 297, 559, 694]
xAxis = [100, 125, 150, 175, 200]

# Plot the first figure
plt.figure()
plt.plot(xAxis, qc, label='QC')
plt.plot(xAxis, hra, label='HRA')
plt.plot(xAxis, dra, label='DRA')

plt.legend()
plt.xlabel('Time Interval (ms)')
plt.ylabel('Satisfied Service Request - SSR')
plt.title('Line Plot of QC, HRA, DRA')
plt.show()

# Data for bar plot
br = np.array([
    [85, 90, 0, 27],
    [92, 93, 0, 33],
    [97, 98, 0, 66],
    [100, 100, 2, 89],
    [100, 100, 54, 98]
])

# Bar plot
fig, ax = plt.subplots()
width = 0.25  # Width of the bars
x = np.arange(len(xAxis))  # X-axis positions

# Plot bars for each series
ax.bar(x - width, br[:, 0], width, label='QC')
ax.bar(x, br[:, 1], width, label='HRA')
ax.bar(x + width, br[:, 2], width, label='DRA')

# Add labels and legend
ax.set_xlabel('Time Interval (ms)')
ax.set_ylabel('Satisfied Service Request - SSR')
ax.set_title('SSR Comparison Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(xAxis)
ax.legend()

# Set axis limits
ax.set_xlim([-0.5, len(xAxis)-0.5])
ax.set_ylim([0, 110])

# Print and save the figure as PDF
plt.savefig('ssr_all_nsf.pdf', orientation='landscape', format='pdf', bbox_inches='tight')

# Show the bar plot
plt.show()

