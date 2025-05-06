import pickle
import numpy as np

# adjust path if needed
pkl = pickle.load(open('proposal_dumps_old.pkl','rb'))

print(f"Number of images: {len(pkl)}")
# inspect first element
first = pkl[0]
print("Keys per entry:", list(first.keys()))
print("  boxes shape:", first['boxes'].shape)
print("  scores shape:", first['scores'].shape)
print("  labels shape:", first['labels'].shape)
# show score stats
print("Scores: min %.4f  max %.4f  mean %.4f" % (
    first['scores'].min(), first['scores'].max(), first['scores'].mean()))

print("\nBox ranges (first 5):")
print(first['boxes'][:5])

x = first['boxes'][:, 0]
y = first['boxes'][:, 1]
z = first['boxes'][:, 2]
print(f"x range: {x.min():.2f} to {x.max():.2f}")
print(f"y range: {y.min():.2f} to {y.max():.2f}")
print(f"z range: {z.min():.2f} to {z.max():.2f}")
