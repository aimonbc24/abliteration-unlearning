"""Summarize LLM-judge unlearning results into honest forget rates.

Each `intervention*-llm-accuracy.csv` reports RETAINED accuracy (fraction of
predictions the Llama-3 judge still marks correct after the intervention; see
utility_scripts/llm_eval.py). Lower is better for unlearning. This script pairs
each intervention treatment with its `baseline` row and reports:

    forget rate (absolute) = baseline_retained - intervention_retained
    forget rate (relative) = (baseline - intervention) / baseline

Run from the repo root:  python analysis/summarize_results.py
"""
import glob
import os

import pandas as pd


def label_for(path: str) -> str:
    p = path.replace("results/", "").replace("-llm-accuracy.csv", "")
    p = p.replace("/intervention", " | ").replace("_results", "")
    return p.replace("/", "/")


def main():
    rows = []
    for f in sorted(glob.glob("results/**/intervention*-llm-accuracy.csv", recursive=True)):
        df = pd.read_csv(f)
        base = df.loc[df["Treatment"] == "baseline", "Accuracy"]
        baseline = float(base.iloc[0]) if len(base) else None
        for _, r in df.iterrows():
            if r["Treatment"] == "baseline":
                continue
            retained = float(r["Accuracy"])
            forget_abs = (baseline - retained) if baseline is not None else None
            forget_rel = (forget_abs / baseline) if baseline else None
            rows.append({
                "dataset / model": label_for(f),
                "setting": r["Treatment"],
                "baseline_retained": baseline,
                "post_retained": retained,
                "forget_abs": forget_abs,
                "forget_rel": forget_rel,
            })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n--- Markdown ---\n")
    print("| dataset / model | setting | baseline retained | post retained | forget rate |")
    print("|---|---|---|---|---|")
    for _, r in out.iterrows():
        print(f"| {r['dataset / model']} | {r['setting']} | "
              f"{r['baseline_retained']:.0%} | {r['post_retained']:.0%} | "
              f"{r['forget_rel']:.0%} |")


if __name__ == "__main__":
    main()
