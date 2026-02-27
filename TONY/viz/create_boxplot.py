from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BoxplotComparer:

    def __init__(self, labels=None, palette=None):
        self.labels = labels or ['Depressed', 'Controls']
        self.palette = palette or ['#A8D5BA', '#6D8A9A']

    def plot(self, data1, data2, filename):
        stat, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        pval_str = self._pval_str(pvalue)

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.set_style("white")

        sns.boxplot(
            data=[data1, data2],
            width=0.45,
            fliersize=0,
            linewidth=1.2,
            palette=self.palette,
            ax=ax
        )
        sns.stripplot(
            data=[data1, data2],
            jitter=0.15,
            alpha=0.4,
            size=4,
            color='#333333',
            ax=ax
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(self.labels, fontsize=11)
        ax.set_ylabel('')
        ax.set_title('')

        all_data = np.concatenate([data1, data2])
        y_max = np.percentile(all_data, 98)
        y_min = min(all_data)

        y_line = y_max * 1.15
        y_text = y_line + y_max * 0.05

        ax.plot([0, 1], [y_line, y_line], color='black', linewidth=1)
        ax.plot([0, 0], [y_line - y_max * 0.02, y_line], color='black', linewidth=1)
        ax.plot([1, 1], [y_line - y_max * 0.02, y_line], color='black', linewidth=1)
        ax.text(0.5, y_text, pval_str, ha='center', va='bottom', fontsize=9)

        ax.set_ylim(y_min - y_max * 0.05, y_text + y_max * 0.08)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()

        return stat, pvalue

    def _pval_str(self, pvalue):
        if pvalue < 0.001:
            return "p < 0.001 ***"
        elif pvalue < 0.01:
            return f"p = {pvalue:.3f} **"
        elif pvalue < 0.05:
            return f"p = {pvalue:.3f} *"
        else:
            return f"p = {pvalue:.3f} (n.s.)"
