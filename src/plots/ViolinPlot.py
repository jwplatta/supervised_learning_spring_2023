from .PlotBase import PlotBase
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ViolinPlot(PlotBase):
    def __init__(self, X):
        if type(X) != pd.DataFrame:
            raise Exception("Expected pd.DataFrame")

        self.X = X
        self.X_scaled = None


    def plot(self, ax=None, figsize=(8, 5)):

        if ax:
            fig = None
        else:
            fig, ax = plt.subplots(figsize=figsize)

        if not(self.X_scaled):
            scaler = StandardScaler()
            self.X_scaled = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)

        positions = range(self.X_scaled.shape[1])

        ax.violinplot(self.X_scaled, positions=positions, showmeans=True, showextrema=True, showmedians=True)

        ax.set_xticks(positions)
        ax.set_xticklabels(self.X_scaled.columns, rotation=45)
        ax.set_title("Feature Variation")
        ax.grid(color='gray', linestyle='--', linewidth=0.2)

        return fig, ax
