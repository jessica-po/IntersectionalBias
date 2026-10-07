from pathlib import Path
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

OUTPUTS_DIR = Path("outputs")
SAMPLE_SIZE = 5
RANDOM_SEED = 42

MODEL_FOLDERS = [
    "llama8B",
    "llama70B",
    "mistral-large-3",
    "gpt5.6",
    "gpt4o",
]

# ============================================================
# SAMPLE FRAMES
# ============================================================

all_samples = []

for folder_name in MODEL_FOLDERS:

    folder = OUTPUTS_DIR / folder_name

    # Find sentence_frames__*.csv automatically
    csv_files = list(folder.glob("sentence_frames__*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No sentence_frames__*.csv found in {folder}"
        )

    if len(csv_files) > 1:
        raise RuntimeError(
            f"More than one sentence_frames CSV found in {folder}: "
            f"{csv_files}"
        )

    csv_path = csv_files[0]

    print(f"\nReading: {csv_path}")

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Keep only structurally valid frames
    # --------------------------------------------------------

    if "auto_structural_valid" not in df.columns:
        raise ValueError(
            f"{csv_path} does not contain auto_structural_valid"
        )

    valid = df[
        df["auto_structural_valid"].astype(str).str.lower() == "true"
    ].copy()

    print(f"  Total frames: {len(df)}")
    print(f"  Structurally valid: {len(valid)}")

    # --------------------------------------------------------
    # Sample 5 frames from every category
    # --------------------------------------------------------

    samples = []

    for category, group in valid.groupby("category"):

        if len(group) < SAMPLE_SIZE:
            raise ValueError(
                f"{folder_name} / {category} only has "
                f"{len(group)} valid frames. Need {SAMPLE_SIZE}."
            )

        sampled = group.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED
        )

        samples.append(sampled)

    model_sample = pd.concat(samples, ignore_index=True)

    # Record where the frame came from
    model_sample["source_folder"] = folder_name

    all_samples.append(model_sample)

    print(
        f"  Selected: {len(model_sample)} "
        f"({model_sample['category'].nunique()} categories × "
        f"{SAMPLE_SIZE})"
    )

# ============================================================
# COMBINE ALL FIVE LLMS
# ============================================================

final_sample = pd.concat(
    all_samples,
    ignore_index=True
)

# Sort so the resulting file is easy to inspect
final_sample = final_sample.sort_values(
    ["category", "source_folder", "frame_id"]
).reset_index(drop=True)

# ============================================================
# SAVE FULL SAMPLE
# ============================================================

final_sample.to_csv(
    "preliminary_sample_300.csv",
    index=False
)

# ============================================================
# ALSO SAVE A CLEANER VERSION FOR MANUAL VALIDATION
# ============================================================

review_columns = [
    "frame_id",
    "source_folder",
    "model_alias",
    "category",
    "category_label",
    "sentence_frame",
    "word_count",
    "human_natural",
    "human_interchangeable",
    "human_neutral",
    "final_accept",
]

review_columns = [
    c for c in review_columns
    if c in final_sample.columns
]

review_df = final_sample[review_columns].copy()

review_df.to_csv(
    "preliminary_sample_300_for_review.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

summary = (
    final_sample
    .groupby(["source_folder", "category"])
    .size()
    .unstack(fill_value=0)
)

print("\n========================================")
print("FINAL SAMPLE")
print("========================================")
print(f"Total frames: {len(final_sample)}")
print(f"Models: {final_sample['source_folder'].nunique()}")
print(f"Categories: {final_sample['category'].nunique()}")

print("\nFrames per model/category:")
print(summary)

print("\nSaved:")
print("  preliminary_sample_300.csv")
print("  preliminary_sample_300_for_review.csv")
