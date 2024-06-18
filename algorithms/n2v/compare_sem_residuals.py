"""
This script was used to generate an image used in the N2V algorithm description,
it uses two trained networks (not included in the repository), one trained normally
and one for a single epoch, on the SEM dataset.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile
from careamics import CAREamist
from careamics_portfolio import PortfolioManager

portfolio = PortfolioManager()

files = portfolio.denoising.N2V_SEM.download()
sem_img = tifffile.imread(files[0])
careamist = CAREamist(Path(__file__).parent / "sem_n2v-trained_long.ckpt")
careamist1e = CAREamist(Path(__file__).parent / "sem_n2v_1epoch.ckpt")

x_s, x_e = 600, 856
y_s, y_e = 200, 456

img = sem_img[y_s:y_e, x_s:x_e]
prediction_1e = careamist1e.predict(source=img).squeeze()
prediction_lt = careamist.predict(source=img).squeeze()
res_1e = img - prediction_1e
res_lt = img - prediction_lt

# plot
fig, ax = plt.subplots(2, 3, figsize=(10, 8))
ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].set_title("Input Image")
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_ylabel("After 1 epoch")

ax[0, 1].imshow(prediction_1e, cmap="gray")
ax[0, 1].set_title("Prediction")
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 2].imshow(res_1e, cmap="gray")
ax[0, 2].set_title("Residuals")
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

ax[1, 0].imshow(img, cmap="gray")
ax[1, 0].set_title("Input Image")
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_ylabel("After 50 epoch")


ax[1, 1].imshow(prediction_lt, cmap="gray")
ax[1, 1].set_title("Prediction")
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
ax[1, 2].imshow(res_lt, cmap="gray")
ax[1, 2].set_title("Residuals")
ax[1, 2].set_xticks([])
ax[1, 2].set_yticks([])


fig.tight_layout()
fig.savefig(Path(__file__).parent / "sem_n2v_residuals_comparison.png")
