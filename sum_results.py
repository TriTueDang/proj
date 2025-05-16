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
    records = []

    for (gender, race, age_decade), matches in stats.items():
        total = len(matches)
        correct = sum(matches)
        accuracy = correct / total if total > 0 else 0
        records.append({
            'Gender': 'Male' if gender == 0 else 'Female',
            'Race': RACE_LABELS.get(race, 'Unknown'),
            'AgeGroup': f"{age_decade*10}s",  # např. "20s", "30s"
            'Accuracy (%)': round(accuracy * 100, 2),
            'Samples': total
        })

    df = pd.DataFrame(records)
    # df = df.reset_index(drop=True)
    print("\n Detection Accuracy Summary:\n")
    print(df.sort_values(['Gender', 'Race', 'AgeGroup']))

    if show_heatmap:
        # Jedna heatmapa pro každé pohlaví
        for gender in df['Gender'].unique():
            subset = df[df['Gender'] == gender]
            pivot = subset.pivot_table(index='Race', columns='AgeGroup', values='Accuracy (%)', aggfunc='mean')

            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f')
            plt.title(f'{name} Detection Accuracy Heatmap – {gender}')
            plt.xlabel('Age Group')
            plt.ylabel('Race')
            plt.show()