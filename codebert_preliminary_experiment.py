#!/usr/bin/env python3
"""
codebert_preliminary_experiment.py

Run the preliminary counterfactual technology-preference experiment with CodeBERT.

INPUT
-----
A CSV already sampled to the frames you want to evaluate, e.g.
    preliminary_sample_300.csv

The CSV must contain:
    category
    sentence_frame

Recommended provenance columns (preserved if present):
    frame_id
    source_folder
    model_alias

EXPERIMENT
----------
For every frame:
  1. Replace {TECH} with every candidate technology for that category.
  2. Score each instantiated candidate with microsoft/codebert-base-mlm.
  3. Use candidate-token length-normalized pseudo-log-likelihood (mean log-probability).
  4. Softmax-normalize scores within the frame/axis.
  5. Rank candidates.
  6. Aggregate by technology and axis.
  7. Run Friedman omnibus tests per axis.
  8. Holm-correct the 12 axis-level p-values.
  9. Run pairwise Wilcoxon signed-rank tests within each axis with Holm correction.

USAGE
-----
    python codebert_preliminary_experiment.py preliminary_sample_300.csv

Optional:
    python codebert_preliminary_experiment.py preliminary_sample_300.csv --outdir codebert_results
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import friedmanchisquare, wilcoxon
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "microsoft/codebert-base-mlm"
SEED = 42

# These must match the categories used in your frame-generation pipeline/paper.
TECH_AXES = {
    "L1_physical": ["USB", "Ethernet", "Bluetooth", "NFC"],
    "L2_data_link": ["HDLC", "Ethernet", "Wi-Fi", "PPP"],
    "L3_network": ["IP", "ICMP", "IGMP", "IPsec"],
    "L4_transport": ["TCP", "UDP", "SCTP", "QUIC"],
    "L5_session": ["NFS", "SMB", "NetBIOS", "RPC"],
    "L6_presentation": ["SSL/TLS", "JPEG", "MPEG", "ASCII"],
    "L7_application": ["HTTP", "FTP", "DNS", "SMTP"],
    "X_language": ["JavaScript", "HTML/CSS", "SQL", "Python", "Bash/Shell"],
    "X_database": ["PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server", "Redis"],
    "X_cloud_development": ["Docker", "npm", "AWS", "pip", "Kubernetes"],
    "X_web_technology": ["Node.js", "React", "jQuery", "Next.js", "Express"],
    "X_ide": ["Visual Studio Code", "Visual Studio", "Notepad++", "IntelliJ IDEA", "Vim"],
}

AXIS_LABELS = {
    "L1_physical": "Physical (L1)",
    "L2_data_link": "Data Link (L2)",
    "L3_network": "Network (L3)",
    "L4_transport": "Transport (L4)",
    "L5_session": "Session (L5)",
    "L6_presentation": "Presentation (L6)",
    "L7_application": "Application (L7)",
    "X_language": "Languages",
    "X_database": "Databases",
    "X_cloud_development": "Cloud Development",
    "X_web_technology": "Web Technologies",
    "X_ide": "Development Environments",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def softmax_np(scores):
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def holm_adjust(pvalues):
    """Holm family-wise error correction."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        corrected = (m - rank) * p[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def bootstrap_mean_ci(values, rng, n_boot=5000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    sample_indices = rng.integers(
        0, len(values), size=(n_boot, len(values))
    )
    boot_means = values[sample_indices].mean(axis=1)

    low = np.quantile(boot_means, alpha / 2)
    high = np.quantile(boot_means, 1 - alpha / 2)

    return float(low), float(high)


class CodeBERTScorer:
    """
    Candidate-token pseudo-log-likelihood scorer.

    For an instantiated candidate, mask each token belonging to the candidate
    one at a time. Score the original token while all other candidate tokens
    remain visible.

    The final candidate score is MEAN token log-probability rather than SUM.
    This reduces the mechanical disadvantage of candidates that tokenize into
    more subword tokens.
    """

    def __init__(self, device: str):
        print(f"Loading {MODEL_NAME}...")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True
        )

        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required because the script uses "
                "character offset mappings to identify candidate tokens."
            )

        if self.tokenizer.mask_token_id is None:
            raise RuntimeError("The selected model has no mask token.")

        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def score_candidate(self, frame: str, candidate: str) -> dict:

        if frame.count("{TECH}") != 1:
            raise ValueError(
                f"Frame must contain exactly one {{TECH}} placeholder:\n{frame}"
            )

        # Character span occupied by the candidate after substitution.
        candidate_start = frame.index("{TECH}")
        sentence = frame.replace("{TECH}", candidate)
        candidate_end = candidate_start + len(candidate)

        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"][0]

        # Find all model tokens whose character spans overlap the inserted
        # candidate's character span.
        candidate_positions = []

        for token_index, (start, end) in enumerate(offsets):
            if start == end:
                # Special tokens such as <s> and </s>.
                continue

            overlaps_candidate = (
                max(start, candidate_start) < min(end, candidate_end)
            )

            if overlaps_candidate:
                candidate_positions.append(token_index)

        if not candidate_positions:
            raise RuntimeError(
                f"Could not identify candidate tokens for {candidate!r} "
                f"in sentence: {sentence}"
            )

        token_logprobs = []
        candidate_tokens = []

        for position in candidate_positions:

            masked_ids = input_ids.clone()
            true_token_id = int(masked_ids[position])

            masked_ids[position] = self.tokenizer.mask_token_id

            model_inputs = {
                "input_ids": masked_ids.unsqueeze(0).to(self.device),
                "attention_mask": encoded["attention_mask"].to(self.device),
            }

            if "token_type_ids" in encoded:
                model_inputs["token_type_ids"] = encoded[
                    "token_type_ids"
                ].to(self.device)

            output = self.model(**model_inputs)

            log_probs = F.log_softmax(
                output.logits[0, position],
                dim=-1
            )

            log_prob = float(
                log_probs[true_token_id].detach().cpu()
            )

            token_logprobs.append(log_prob)

            candidate_tokens.append(
                self.tokenizer.convert_ids_to_tokens(true_token_id)
            )

        sum_logprob = float(np.sum(token_logprobs))
        mean_logprob = float(np.mean(token_logprobs))

        return {
            "sentence": sentence,
            "candidate_token_count": len(candidate_positions),
            "candidate_tokens": json.dumps(
                candidate_tokens,
                ensure_ascii=False
            ),
            "sum_logprob": sum_logprob,
            "mean_logprob": mean_logprob,
            "pseudo_perplexity": float(math.exp(-mean_logprob)),
        }


def validate_input(df: pd.DataFrame):
    required = {"category", "sentence_frame"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}"
        )

    if len(df) == 0:
        raise ValueError("Input CSV contains no rows.")

    unknown_categories = sorted(
        set(df["category"].dropna()) - set(TECH_AXES)
    )

    if unknown_categories:
        raise ValueError(
            "The CSV contains categories that are not defined in TECH_AXES:\n"
            + "\n".join(unknown_categories)
        )

    invalid_placeholder = (
        df["sentence_frame"]
        .astype(str)
        .map(lambda x: x.count("{TECH}") != 1)
    )

    if invalid_placeholder.any():
        bad = df.loc[
            invalid_placeholder,
            ["category", "sentence_frame"]
        ]

        raise ValueError(
            "Some sampled frames do not contain exactly one {TECH}:\n"
            + bad.head(10).to_string(index=False)
        )


def run_experiment(
    df: pd.DataFrame,
    scorer: CodeBERTScorer
) -> pd.DataFrame:

    rows = []

    # Number of candidate evaluations, not number of model forward passes.
    total_candidates = sum(
        len(TECH_AXES[category])
        for category in df["category"]
    )

    progress = tqdm(
        total=total_candidates,
        desc="Scoring candidates"
    )

    for row_number, row in df.iterrows():

        category = row["category"]
        frame = str(row["sentence_frame"])
        candidates = TECH_AXES[category]

        frame_id = (
            row["frame_id"]
            if "frame_id" in df.columns
            else f"sample_{row_number:04d}"
        )

        source_generator = ""

        if "source_folder" in df.columns:
            source_generator = row["source_folder"]
        elif "model_alias" in df.columns:
            source_generator = row["model_alias"]

        candidate_records = []
        scores = []

        for candidate in candidates:

            result = scorer.score_candidate(
                frame=frame,
                candidate=candidate
            )

            record = {
                "frame_id": frame_id,
                "source_generator": source_generator,
                "category": category,
                "axis_label": AXIS_LABELS[category],
                "frame": frame,
                "candidate": candidate,
                **result,
            }

            candidate_records.append(record)
            scores.append(result["mean_logprob"])

            progress.update(1)

        # Normalize only among the candidates belonging to this exact axis/frame.
        probabilities = softmax_np(scores)

        # Higher log-probability = stronger preference.
        order = np.argsort(-np.asarray(scores))

        ranks = np.empty(len(candidates), dtype=int)

        for rank, candidate_index in enumerate(order, start=1):
            ranks[candidate_index] = rank

        entropy = float(
            -np.sum(
                probabilities *
                np.log(probabilities + 1e-12)
            )
        )

        maximum_entropy = math.log(len(candidates))
        normalized_entropy = entropy / maximum_entropy

        best_score = max(scores)

        for i, record in enumerate(candidate_records):

            record["probability_within_axis"] = float(
                probabilities[i]
            )

            record["rank_within_frame"] = int(ranks[i])

            record["is_top_choice"] = bool(
                ranks[i] == 1
            )

            record["delta_from_best_logscore"] = float(
                scores[i] - best_score
            )

            record["frame_entropy"] = entropy
            record["normalized_frame_entropy"] = (
                normalized_entropy
            )

            rows.append(record)

    progress.close()

    return pd.DataFrame(rows)


def preference_summary(raw: pd.DataFrame) -> pd.DataFrame:

    rng = np.random.default_rng(SEED)

    rows = []

    grouped = raw.groupby(
        ["category", "axis_label", "candidate"]
    )

    for (category, axis_label, candidate), group in grouped:

        probability_low, probability_high = bootstrap_mean_ci(
            group["probability_within_axis"].to_numpy(),
            rng
        )

        top_low, top_high = bootstrap_mean_ci(
            group["is_top_choice"]
            .astype(float)
            .to_numpy(),
            rng
        )

        rows.append({
            "category": category,
            "axis_label": axis_label,
            "candidate": candidate,
            "n_frames": group["frame_id"].nunique(),

            "mean_probability":
                group["probability_within_axis"].mean(),

            "probability_ci_low": probability_low,
            "probability_ci_high": probability_high,

            "top_choice_count":
                int(group["is_top_choice"].sum()),

            "top_choice_rate":
                group["is_top_choice"].mean(),

            "top_choice_ci_low": top_low,
            "top_choice_ci_high": top_high,

            "mean_rank":
                group["rank_within_frame"].mean(),

            "mean_logprob":
                group["mean_logprob"].mean(),

            "mean_delta_from_best_logscore":
                group["delta_from_best_logscore"].mean(),

            "mean_candidate_token_count":
                group["candidate_token_count"].mean(),
        })

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        ["category", "mean_rank", "mean_probability"],
        ascending=[True, True, False]
    ).reset_index(drop=True)


def axis_tests(raw: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for (category, axis_label), group in raw.groupby(
        ["category", "axis_label"]
    ):

        pivot = group.pivot_table(
            index="frame_id",
            columns="candidate",
            values="mean_logprob",
            aggfunc="first"
        )

        # Only complete matched frames should enter the Friedman test.
        pivot = pivot.dropna()

        candidates = [
            candidate
            for candidate in TECH_AXES[category]
            if candidate in pivot.columns
        ]

        n_frames = len(pivot)
        n_candidates = len(candidates)

        if n_frames >= 2 and n_candidates >= 3:

            statistic, p_value = friedmanchisquare(
                *[
                    pivot[candidate].to_numpy()
                    for candidate in candidates
                ]
            )

            kendalls_w = (
                statistic /
                (n_frames * (n_candidates - 1))
            )

        else:
            statistic = np.nan
            p_value = np.nan
            kendalls_w = np.nan

        mean_normalized_entropy = (
            group.groupby("frame_id")[
                "normalized_frame_entropy"
            ]
            .first()
            .mean()
        )

        rows.append({
            "category": category,
            "axis_label": axis_label,
            "n_frames": n_frames,
            "n_candidates": n_candidates,
            "friedman_chi2": statistic,
            "friedman_p": p_value,
            "kendalls_w": kendalls_w,
            "mean_normalized_entropy":
                mean_normalized_entropy,
            "preference_concentration":
                1 - mean_normalized_entropy,
        })

    results = pd.DataFrame(rows)

    valid = results["friedman_p"].notna()

    results["friedman_p_holm"] = np.nan

    if valid.any():
        results.loc[
            valid,
            "friedman_p_holm"
        ] = holm_adjust(
            results.loc[
                valid,
                "friedman_p"
            ].tolist()
        )

    results["significant_0_05"] = (
        results["friedman_p_holm"] < 0.05
    )

    return results


def pairwise_tests(raw: pd.DataFrame) -> pd.DataFrame:

    all_results = []

    for (category, axis_label), group in raw.groupby(
        ["category", "axis_label"]
    ):

        pivot = group.pivot_table(
            index="frame_id",
            columns="candidate",
            values="mean_logprob",
            aggfunc="first"
        ).dropna()

        candidates = [
            candidate
            for candidate in TECH_AXES[category]
            if candidate in pivot.columns
        ]

        axis_results = []

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):

                candidate_a = candidates[i]
                candidate_b = candidates[j]

                scores_a = pivot[candidate_a].to_numpy()
                scores_b = pivot[candidate_b].to_numpy()

                differences = scores_a - scores_b

                if np.allclose(differences, 0):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    statistic, p_value = wilcoxon(
                        scores_a,
                        scores_b,
                        alternative="two-sided",
                        zero_method="wilcox"
                    )

                axis_results.append({
                    "category": category,
                    "axis_label": axis_label,
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "n_frames": len(pivot),

                    "mean_logprob_difference_a_minus_b":
                        float(np.mean(differences)),

                    "median_logprob_difference_a_minus_b":
                        float(np.median(differences)),

                    "wilcoxon_stat":
                        float(statistic),

                    "wilcoxon_p":
                        float(p_value),
                })

        if axis_results:

            corrected = holm_adjust(
                [
                    result["wilcoxon_p"]
                    for result in axis_results
                ]
            )

            for result, adjusted_p in zip(
                axis_results,
                corrected
            ):

                result["wilcoxon_p_holm"] = adjusted_p
                result["significant_0_05"] = (
                    adjusted_p < 0.05
                )

                all_results.append(result)

    return pd.DataFrame(all_results)


def generator_robustness(raw: pd.DataFrame) -> pd.DataFrame:

    if (
        "source_generator" not in raw.columns
        or raw["source_generator"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .all()
    ):
        return pd.DataFrame()

    results = (
        raw.groupby(
            [
                "category",
                "axis_label",
                "source_generator",
                "candidate"
            ],
            as_index=False
        )
        .agg(
            n_frames=("frame_id", "nunique"),

            mean_probability=(
                "probability_within_axis",
                "mean"
            ),

            top_choice_rate=(
                "is_top_choice",
                "mean"
            ),

            mean_rank=(
                "rank_within_frame",
                "mean"
            )
        )
    )

    return results.sort_values(
        [
            "category",
            "source_generator",
            "mean_rank",
            "mean_probability"
        ],
        ascending=[True, True, True, False]
    )


def axis_winners(summary: pd.DataFrame) -> pd.DataFrame:

    winners = (
        summary
        .sort_values(
            [
                "category",
                "mean_rank",
                "mean_probability"
            ],
            ascending=[True, True, False]
        )
        .groupby(
            "category",
            as_index=False
        )
        .first()
    )

    return winners[[
        "category",
        "axis_label",
        "candidate",
        "n_frames",
        "mean_probability",
        "probability_ci_low",
        "probability_ci_high",
        "top_choice_count",
        "top_choice_rate",
        "mean_rank",
    ]].rename(
        columns={
            "candidate":
                "most_preferred_candidate"
        }
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_csv",
        help="Already-sampled CSV, e.g. preliminary_sample_300.csv"
    )

    parser.add_argument(
        "--outdir",
        default="codebert_results",
        help="Directory where result files will be written"
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps"
        ],
        default="auto"
    )

    args = parser.parse_args()

    set_seed(SEED)

    input_path = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    validate_input(df)

    # Create a unique internal ID in case frame_id values repeat between
    # the five original generator files.
    df = df.copy()

    df["_experiment_frame_id"] = [
        f"F{i:04d}"
        for i in range(1, len(df) + 1)
    ]

    if "frame_id" in df.columns:
        df["original_frame_id"] = df["frame_id"]

    df["frame_id"] = df["_experiment_frame_id"]

    print("\n==========================================")
    print("CODEBERT PRELIMINARY EXPERIMENT")
    print("==========================================")
    print(f"Input file: {input_path}")
    print(f"Frames: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")

    if "source_folder" in df.columns:
        print(
            f"Frame generators: "
            f"{df['source_folder'].nunique()}"
        )

    print("\nFrames per category:")
    print(
        df["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nNo frames will be sampled or removed.")
    print("Every row in the input CSV will be evaluated.")

    device = choose_device(args.device)

    print(f"\nDevice: {device}")
    print(f"Model: {MODEL_NAME}")

    scorer = CodeBERTScorer(device)

    raw = run_experiment(
        df=df,
        scorer=scorer
    )

    # ---------------------------------------------------------
    # Save raw candidate-level scores
    # ---------------------------------------------------------

    raw.to_csv(
        outdir / "raw_candidate_scores.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Descriptive preference summary
    # ---------------------------------------------------------

    summary = preference_summary(raw)

    summary.to_csv(
        outdir / "preference_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Axis-level significance tests
    # ---------------------------------------------------------

    tests = axis_tests(raw)

    tests.to_csv(
        outdir / "axis_omnibus_tests.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Pairwise follow-up tests
    # ---------------------------------------------------------

    pairwise = pairwise_tests(raw)

    pairwise.to_csv(
        outdir / "pairwise_wilcoxon_tests.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Robustness by original frame generator
    # ---------------------------------------------------------

    robustness = generator_robustness(raw)

    if not robustness.empty:
        robustness.to_csv(
            outdir / "generator_robustness.csv",
            index=False
        )

    # ---------------------------------------------------------
    # Simple paper-friendly winners table
    # ---------------------------------------------------------

    winners = axis_winners(summary)

    winners = winners.merge(
        tests[[
            "category",
            "friedman_chi2",
            "friedman_p_holm",
            "kendalls_w",
            "preference_concentration",
            "significant_0_05"
        ]],
        on="category",
        how="left"
    )

    winners.to_csv(
        outdir / "axis_winners.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Run metadata
    # ---------------------------------------------------------

    metadata = {
        "model": MODEL_NAME,
        "seed": SEED,
        "input_csv": str(input_path),
        "number_of_frames": len(df),
        "number_of_categories":
            int(df["category"].nunique()),
        "scoring_method": (
            "candidate-token pseudo-log-likelihood; "
            "one candidate token masked at a time; "
            "mean token log-probability used as "
            "length-normalized candidate score"
        ),
        "within_frame_normalization":
            "softmax across candidates in the same axis",
        "axis_candidates": TECH_AXES,
        "statistical_tests": {
            "omnibus":
                "Friedman test across matched frame scores",
            "axis_multiple_testing":
                "Holm correction across 12 axis-level tests",
            "effect_size":
                "Kendall's W",
            "pairwise":
                "Wilcoxon signed-rank tests with Holm correction"
        }
    }

    (
        outdir /
        "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print("\n==========================================")
    print("MOST-PREFERRED CANDIDATE PER AXIS")
    print("==========================================")

    display_columns = [
        "axis_label",
        "most_preferred_candidate",
        "n_frames",
        "mean_probability",
        "top_choice_rate",
        "mean_rank",
        "friedman_p_holm",
        "kendalls_w",
        "significant_0_05"
    ]

    print(
        winners[display_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n==========================================")
    print("DONE")
    print("==========================================")
    print(f"Results saved to: {outdir.resolve()}")

    print("\nMain files to inspect:")
    print("  1. axis_winners.csv")
    print("  2. axis_omnibus_tests.csv")
    print("  3. preference_summary.csv")
    print("  4. pairwise_wilcoxon_tests.csv")
    print("  5. generator_robustness.csv")
    print("  6. raw_candidate_scores.csv")


if __name__ == "__main__":
    main()
