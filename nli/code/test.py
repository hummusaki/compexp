
import pandas as pd

path = "exp/snli_1.0_dev-6-sentence-5/preds_acts.csv"
df = pd.read_csv(path)

counts = {
    int(col): int(df[col].sum())
    for col in df.columns if col.isdigit()
}

print("Original neuron activation counts:")
for n in range(10):
    print(f"Neuron {n}: {counts[n]}")

eligible = [
    n for n, count in sorted(counts.items())
    if count >= 500
]

print("\nTen suggested additional neurons:")
print(eligible[:10])
