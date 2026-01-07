### Author: OptimusThi
"""
Python Script to generate the paper figures
"""
import matplotlib.pyplot as plt
import scienceplots


# Python sanity check to ensure that all figures are in the PRC requirements!
for text in fig.findobj(match=plt.Text):
    if text.get_fontsize() < MIN_FONT_PT:
        print("Warning:", text.get_text())
