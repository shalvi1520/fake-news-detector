"""
Exploratory Data Analysis — Fake News Dataset
Run: python notebooks/eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# ── Load Data ────────────────────────────────────────────────
fake_df = pd.read_csv("data/raw/Fake.csv")
true_df = pd.read_csv("data/raw/True.csv")

# Add labels
fake_df["label"] = 0   # 0 = Fake
true_df["label"] = 1   # 1 = Real

# Combine
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Total samples  : {len(df)}")
print(f"Fake articles  : {len(df[df['label'] == 0])}")
print(f"Real articles  : {len(df[df['label'] == 1])}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── Class Distribution ─────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=df, palette=["#e74c3c", "#2ecc71"])
plt.xticks([0, 1], ["Fake", "Real"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("assets/class_distribution.png", dpi=150)
plt.show()
print(" Saved: assets/class_distribution.png")

# ── Text Length Distribution ───────────────────────────────
df["text_length"] = df["text"].astype(str).apply(len)

plt.figure(figsize=(10, 4))
sns.histplot(data=df, x="text_length", hue="label", bins=50, 
             palette=["#e74c3c", "#2ecc71"], alpha=0.6)
plt.title("Article Text Length Distribution")
plt.xlabel("Character Count")
plt.xlim(0, 20000)
plt.tight_layout()
plt.savefig("assets/text_length_dist.png", dpi=150)
plt.show()
print(" Saved: assets/text_length_dist.png")

# ── Subject Distribution ───────────────────────────────────
plt.figure(figsize=(12, 4))
fake_subjects = fake_df["subject"].value_counts()
true_subjects = true_df["subject"].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fake_subjects.plot(kind="bar", ax=ax1, color="#e74c3c", title="Fake News — Subjects")
true_subjects.plot(kind="bar", ax=ax2, color="#2ecc71", title="Real News — Subjects")
plt.tight_layout()
plt.savefig("assets/subject_distribution.png", dpi=150)
plt.show()
print(" Saved: assets/subject_distribution.png")

# ── Word Cloud ──────────────────────────────────────────────
fake_text = " ".join(df[df["label"] == 0]["text"].astype(str).tolist())
real_text = " ".join(df[df["label"] == 1]["text"].astype(str).tolist())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

wc_fake = WordCloud(width=800, height=400, background_color="white",
                    colormap="Reds", max_words=100).generate(fake_text[:100000])
wc_real = WordCloud(width=800, height=400, background_color="white",
                    colormap="Greens", max_words=100).generate(real_text[:100000])

ax1.imshow(wc_fake, interpolation="bilinear")
ax1.set_title("Fake News — Most Common Words", fontsize=14)
ax1.axis("off")

ax2.imshow(wc_real, interpolation="bilinear")
ax2.set_title("Real News — Most Common Words", fontsize=14)
ax2.axis("off")

plt.tight_layout()
plt.savefig("assets/wordclouds.png", dpi=150)
plt.show()
print(" Saved: assets/wordclouds.png")

print("\n EDA Complete!")