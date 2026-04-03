import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RACE_LABELS = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Other'
}
def summarize_results(stats, name, show_heatmap=True):
    AGE_GROUPS_ORDER = ["0-2", "3-5", "6-12", "13-18", "19-25", "26-35", "36-50", "51-70", "71-90", "90+"]
    age_sort_map = {name: i for i, name in enumerate(AGE_GROUPS_ORDER)}

    records = []
    for (gender, race, age_group), matches in stats.items():
        total = len(matches)
        correct = sum(matches)
        accuracy = correct / total if total > 0 else 0
        records.append({
            'Gender': 'Male' if gender == 0 else 'Female',
            'Race': RACE_LABELS.get(race, 'Unknown'),
            'AgeGroup': age_group,
            'Accuracy (%)': round(accuracy * 100, 2),
            'Samples': total
        })

    pd.set_option('display.max_rows', None)  # Show all rows
    df = pd.DataFrame(records)
    df['AgeSort'] = df['AgeGroup'].map(age_sort_map)
    df_sorted = df.sort_values(['Gender', 'Race', 'AgeSort']).reset_index(drop=True)
    df_sorted = df_sorted.drop(columns=['AgeSort'])

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{name}_detection_accuracy_summary.csv"

    df_sorted.to_csv(filename, index=False)
    print("\n Detection Accuracy Summary:\n")
    print(df_sorted)

    if show_heatmap:
        # Jedna heatmapa pro každé pohlaví
        for gender in df_sorted['Gender'].unique():
            subset = df_sorted[df_sorted['Gender'] == gender]
            pivot = subset.pivot_table(index='Race', columns='AgeGroup', values='Accuracy (%)', aggfunc='mean')

            # Reorder columns according to AGE_GROUPS_ORDER
            existing_cols = [c for c in AGE_GROUPS_ORDER if c in pivot.columns]
            pivot = pivot[existing_cols]

            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f')
            plt.title(f'{name} Detection Accuracy Heatmap – {gender}')
            plt.xlabel('Age Group')
            plt.ylabel('Race')
            plt.show()

