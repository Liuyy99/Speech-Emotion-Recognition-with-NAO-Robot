import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import cPickle
import logging
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


f = open('./zscore_all_datasets_combined_SI_Valid_SI_Test40.pkl', 'rb')
mean1, std1, mean2, std2, mean3, std3 = cPickle.load(f)

x = np.arange(0, 40, 1)
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (5,10))
fig.tight_layout(pad=0.4)
datasets_index = ["DG", "DU", "DR", "DS", "DC", "DI"]
datasets_name = {
    "DG": "EmoDB", 
    "DU": "Urdu", 
    "DR": "RAVDESS", 
    "DS": "SAVEE", 
    "DC": "CASIA", 
    "DI": "IEMOCAP"
}
for index in datasets_index:
    ax1.plot(x, mean1[index], label=datasets_name[index])
#     ax1.set_xlabel("Dimension (Filter Bank)")
#     ax1.set_ylabel("Value (Mel)")
#     ax1.set_title('Neutral Mean of Static Log-Mel')
ax1.legend() 

for index in datasets_index:
    ax2.plot(x, mean2[index], label=datasets_name[index])
#     ax2.set_xlabel("Dimension (Filter Bank)")
#     ax2.set_ylabel("Value (Delta Mel)")
#     ax2.set_title('Neutral Mean of Delta Log-Mel')
# ax2.legend()

for index in datasets_index:
    ax3.plot(x, mean3[index], label=datasets_name[index])
#     ax3.set_xlabel("Dimension (Filter Bank)")
#     ax3.set_ylabel("Value (Delta-delta Mel)")
#     ax3.set_title('Neutral Mean of Delta-delta Log-Mel')
# ax3.legend()

fig_name = "Neural_mean_of_6_datasets.png"
plt.savefig(fig_name)
