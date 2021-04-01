from matplotlib import pyplot as plt
import pandas as pd

metrics_file = r"C:\Users\bciap\Desktop\Res\metrics.csv"

df_metrics = pd.read_csv(metrics_file)

plt.boxplot (df_metrics.test_acc)
plt.title ("Test accuracy 200 class classification")
#plt.title ("20pt Affine + Baseline aug; 10 experiments")