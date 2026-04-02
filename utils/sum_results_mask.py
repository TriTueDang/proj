import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_mask_results(stats, name, show_plot=True):
    """
    Process raw Face-Mask validation data (TP, FP, FN for a detector)
    and compute Precision, Recall, and Accuracy. Saves results to CSV and shows a bar chart.

    If category-level stats are provided, separate charts are shown for
    with_mask, incorrect_mask, and without_mask.

    :param stats: Either a dict with TP/FP/FN or a category-level dict with values TP/FP/FN.
    :param name: Detector name (e.g. 'HAAR')
    """
    category_records = None
    if isinstance(stats, dict) and stats and all(isinstance(v, dict) for v in stats.values()):
        category_records = []
        for category, values in stats.items():
            if category == 'background':
                continue
            tp = values.get('TP', 0)
            fp = values.get('FP', 0)
            fn = values.get('FN', 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            category_records.append({
                'MaskStatus': category,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Precision (%)': round(precision * 100, 2),
                'Recall (%)': round(recall * 100, 2),
                'Accuracy (%)': round(accuracy * 100, 2)
            })

        print(f"\nCategory-level results for {name}:")
        print(pd.DataFrame(category_records).to_string(index=False))

        TP = sum(r['TP'] for r in category_records)
        FP = sum(r['FP'] for r in category_records)
        FN = sum(r['FN'] for r in category_records)
    else:
        TP = stats.get('TP', 0)
        FP = stats.get('FP', 0)
        FN = stats.get('FN', 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    records = [{
        'Detector': name,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision (%)': round(precision * 100, 2),
        'Recall (%)': round(recall * 100, 2),
        'Accuracy (%)': round(accuracy * 100, 2)
    }]

    # Create a DataFrame
    df = pd.DataFrame(records)

    # Save results to CSV
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    out_csv = os.path.join(results_dir, f"{name}_face_mask_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n Detection Accuracy Summary for {name}:\n")
    print(df.to_string(index=False))
    print(f"--> Uloženo do: {out_csv}")

    if show_plot:
        if category_records:
            # Show separate bar charts for each mask status category.
            fig, axes = plt.subplots(1, len(category_records), figsize=(5 * len(category_records), 5), sharey=True)
            if len(category_records) == 1:
                axes = [axes]

            for ax, record in zip(axes, category_records):
                metrics = {
                    'Precision (%)': record['Precision (%)'],
                    'Recall (%)': record['Recall (%)'],
                    'Accuracy (%)': record['Accuracy (%)']
                }
                sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette=['#1f77b4', '#ff7f0e', '#2ca02c'], ax=ax)
                ax.set_title(f"{name} - {record['MaskStatus']}")
                ax.set_ylim(0, 100)
                ax.set_ylabel('Percent (%)' if ax is axes[0] else '')
                for i, v in enumerate(metrics.values()):
                    ax.text(i, v + 2, f"{v:.1f}%", color='black', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.show()

            # Also show a grouped chart comparing all mask status categories.
            category_df = pd.DataFrame(category_records)
            grouped_df = category_df.melt(
                id_vars='MaskStatus',
                value_vars=['Precision (%)', 'Recall (%)', 'Accuracy (%)'],
                var_name='Metric',
                value_name='Value'
            )
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=grouped_df,
                x='Metric',
                y='Value',
                hue='MaskStatus',
                palette=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            plt.title(f'{name} Performance by Mask Status')
            plt.ylabel('Percent (%)')
            plt.ylim(0, 100)
            plt.legend(title='Mask Status')
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height + 1,
                    f"{height:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            plt.tight_layout()
            plt.show()
        else:
            metrics = {'Precision (%)': precision * 100, 'Recall (%)': recall * 100, 'Accuracy (%)': accuracy * 100}
            plt.figure(figsize=(6, 4))

            # Use a color palette for distinct bar colors
            ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
            plt.title(f'{name} Detection Performance on Face-Mask')
            plt.ylabel('Percent (%)')
            plt.ylim(0, 100)

            # Add labels above each bar
            for i, v in enumerate(metrics.values()):
                ax.text(i, v + 2, f"{v:.1f}%", color='black', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.show()
