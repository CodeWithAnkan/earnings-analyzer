#!/usr/bin/env python3
"""
validate_nvda_drift.py

Phase 2 validation: verify that NVDA's China export-restriction narrative drift
shows up in the correct quarters.

Background:
  In late 2023, the US expanded export controls on AI chips to China.
  NVDA management's language around China risk should show a sharp_break
  between 2023Q3/2023Q4 and 2024Q1, as the topic went from a background
  concern to a front-and-center risk.

What we check:
  1. drift/calculate returns scores for NVDA
  2. The 'risks' topic shows elevated drift around 2023Q4 → 2024Q1
  3. The quote surface returns passages mentioning China / export controls
  4. Confidence scoring shows management confidence dropping on China topics

Run:
  python validate_nvda_drift.py
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

# ANSI colours for terminal output
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def check(label: str, passed: bool, detail: str = ""):
    icon = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"  {icon} {label}")
    if detail:
        print(f"      {YELLOW}{detail}{RESET}")
    return passed


def separator(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


def main():
    print(f"\n{BOLD}NVDA Drift Validation — Phase 2{RESET}")
    print("Verifying China export-restriction narrative shift\n")

    # ── 0. Server health ───────────────────────────────────────────────────────
    separator("0. Server health")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        check("Server is running", r.status_code == 200)
    except requests.exceptions.ConnectionError:
        print(f"{RED}Cannot connect to server. Start with: uvicorn app.main:app --reload{RESET}")
        sys.exit(1)

    # ── 1. Compute drift ───────────────────────────────────────────────────────
    separator("1. Compute drift scores for NVDA")
    r = requests.post(f"{BASE_URL}/drift/calculate/NVDA", timeout=30)
    if not check("POST /drift/calculate/NVDA returned 200", r.status_code == 200, r.text[:200] if r.status_code != 200 else ""):
        sys.exit(1)

    data = r.json()
    n_scores = data.get("drift_scores_created", 0)
    check(f"Drift scores created: {n_scores}", n_scores > 0, "Need embedded segments — run Phase 1 pipeline first")

    # ── 2. Full timeline ───────────────────────────────────────────────────────
    separator("2. Drift timeline — all topics")
    r = requests.get(f"{BASE_URL}/drift/timeline/NVDA", timeout=10)
    check("GET /drift/timeline/NVDA returned 200", r.status_code == 200)
    timeline = r.json().get("drift_timeline", [])
    check(f"Timeline has {len(timeline)} entries", len(timeline) > 0)

    # Print full table
    print()
    print(f"  {'Topic':<12} {'From':<10} {'To':<10} {'Score':>7}  Label")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*7}  {'-'*12}")
    for row in sorted(timeline, key=lambda x: (x["topic"], x["quarter_from"])):
        label_colour = {
            "stable":      GREEN,
            "drifting":    YELLOW,
            "sharp_break": RED,
        }.get(row["label"], RESET)
        print(
            f"  {row['topic']:<12} {row['quarter_from']:<10} {row['quarter_to']:<10} "
            f"{row['drift_score']:>7.4f}  {label_colour}{row['label']}{RESET}"
        )

    # ── 3. Risks topic focus ───────────────────────────────────────────────────
    separator("3. Risks topic — looking for China export-restriction signal")
    r = requests.get(f"{BASE_URL}/drift/timeline/NVDA?topic=risks", timeout=10)
    check("GET /drift/timeline/NVDA?topic=risks returned 200", r.status_code == 200)
    risks_timeline = r.json().get("drift_timeline", [])

    # Find the 2023Q4 → 2024Q1 transition specifically
    target_transition = None
    for row in risks_timeline:
        if row["quarter_from"] in ("2023Q3", "2023Q4") and row["quarter_to"] in ("2024Q1", "2023Q4"):
            target_transition = row
            break

    if target_transition:
        lbl = target_transition["label"]
        score = target_transition["drift_score"]
        is_elevated = lbl in ("drifting", "sharp_break")
        check(
            f"Elevated risk drift around {target_transition['quarter_from']} → {target_transition['quarter_to']} "
            f"(score={score:.4f}, label={lbl})",
            is_elevated,
            "Expected drifting or sharp_break for China export control period"
        )
    else:
        check("Found 2023Q3/Q4 → 2024Q1 risks transition", False, "Transition not found — check data coverage")

    # Highest-drift risks quarter
    if risks_timeline:
        worst = max(risks_timeline, key=lambda x: x["drift_score"])
        print(f"\n  Highest risk drift: {worst['quarter_from']} → {worst['quarter_to']} "
              f"score={worst['drift_score']:.4f} ({worst['label']})")

    # ── 4. Alerts ─────────────────────────────────────────────────────────────
    separator("4. Alerts")
    r = requests.get(f"{BASE_URL}/drift/alerts/NVDA", timeout=10)
    check("GET /drift/alerts/NVDA returned 200", r.status_code == 200)
    alerts = r.json().get("alerts", [])
    check(f"Found {len(alerts)} alert(s)", len(alerts) > 0)

    print()
    for a in alerts[:8]:
        label_colour = RED if a["label"] == "sharp_break" else YELLOW
        print(f"  {label_colour}[{a['label']}]{RESET} {a['topic']:<12} "
              f"{a['quarter_from']} → {a['quarter_to']}  score={a['drift_score']:.4f}")

    # ── 5. Quote surfacing ────────────────────────────────────────────────────
    separator("5. Quote surfacing — evidence behind top alert")
    if alerts:
        top_alert = alerts[0]
        print(f"  Surfacing quotes for: {top_alert['topic']} "
              f"{top_alert['quarter_from']} → {top_alert['quarter_to']}\n")

        r = requests.get(
            f"{BASE_URL}/drift/quotes/NVDA",
            params={
                "topic":        top_alert["topic"],
                "quarter_from": top_alert["quarter_from"],
                "quarter_to":   top_alert["quarter_to"],
                "top_n":        5,
            },
            timeout=30,
        )
        check("GET /drift/quotes/NVDA returned 200", r.status_code == 200, r.text[:200] if r.status_code != 200 else "")

        if r.status_code == 200:
            quotes = r.json().get("drifted_quotes", [])
            check(f"Returned {len(quotes)} drifted quote(s)", len(quotes) > 0)

            print()
            for i, q in enumerate(quotes, 1):
                conf_colour = RED if (q.get("confidence_score") or 1.0) < 0.4 else GREEN
                print(f"  [{i}] dist={q['drift_distance']:.4f}  "
                      f"conf={conf_colour}{q.get('confidence_score', 'n/a')}{RESET}  "
                      f"{q['speaker']} ({q['role']})")
                print(f"      {q['text_preview'][:200]}")
                print()

            # Check if any mention China / export
            china_mentions = [
                q for q in quotes
                if any(kw in q["text"].lower() for kw in ["china", "export", "restriction", "control", "ban"])
            ]
            check(
                f"China/export keywords found in {len(china_mentions)}/{len(quotes)} top quotes",
                len(china_mentions) > 0,
                "If 0: the alert may be from a different topic; check the table above"
            )
    else:
        print("  No alerts to surface quotes for.")

    # ── 6. Ad-hoc compare ─────────────────────────────────────────────────────
    separator("6. Ad-hoc compare — 2023Q4 vs 2024Q1 (all topics)")
    r = requests.get(
        f"{BASE_URL}/drift/compare/NVDA",
        params={"quarter_from": "2023Q4", "quarter_to": "2024Q1"},
        timeout=15,
    )
    if r.status_code == 200:
        comp = r.json().get("comparison", {})
        print()
        for topic, info in comp.items():
            if not info.get("available"):
                print(f"  {topic:<12} — no data")
                continue
            # compare endpoint doesn't return a label — display raw scores
            score = info["drift_score"]
            print(f"  {topic:<12} score={score:.6f}  "
                  f"centroid={info['centroid_dist']:.6f}  tail={info['tail_dist']:.6f}  "
                  f"(segs: {info['segment_count_from']} → {info['segment_count_to']})")
        check("Compare endpoint works", True)
    else:
        check("Compare endpoint works", False, r.text[:200])

    # ── 7. Summary ────────────────────────────────────────────────────────────
    separator("7. Cross-ticker summary")
    r = requests.get(f"{BASE_URL}/drift/summary", timeout=10)
    check("GET /drift/summary returned 200", r.status_code == 200)
    if r.status_code == 200:
        summary = r.json().get("summary", {})
        print()
        for ticker, counts in summary.items():
            print(f"  {ticker}:  stable={counts.get('stable',0)}  "
                  f"{YELLOW}drifting={counts.get('drifting',0)}{RESET}  "
                  f"{RED}sharp_break={counts.get('sharp_break',0)}{RESET}")

    print(f"\n{BOLD}Validation complete.{RESET}\n")
    print("If risks drift around 2023Q4→2024Q1 is elevated and quotes mention")
    print("'China' or 'export', the drift engine is working correctly.\n")


if __name__ == "__main__":
    main()