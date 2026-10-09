# AGENTS.md — SERF Research AI Operating Constraints

> **Status:** Mandatory operating policy for AI agents and scheduled ChatGPT sessions working in this repository.
> **Project:** `egbertbritannia-cpu/rbl-requirements-classification`
> **Owner / decision maker:** Quách Thế Dương (single-person research project)
> **Created:** 2026-10-09
> **Scope:** Repository root and all subdirectories.
> **Language:** Use Vietnamese for progress reports and tracking notes; preserve technical terms, code, citations, and the manuscript's English unless explicitly directed otherwise.

## 0. Mission and decision hierarchy

The goal is to investigate and, **only if supported by evidence**, address the scientific and methodological weaknesses identified after the rejection of the SERF paper at EAI FISAT 2026.

Order of priorities:

1. **Scientific integrity and reproducibility** over attractive numbers, novelty claims, or apparent task completion.
2. **Protection of existing research data, checkpoints, manuscripts, and experiment outputs** over convenience.
3. **Directly testing reviewer criticisms** over introducing speculative new features.
4. **Small, auditable, reversible changes** over broad refactors.
5. **Accurate updates to the canonical Google Sheet** over maintaining duplicate progress documents.

Follow the current explicit user request when it is compatible with safety and scientific integrity. If a requested action is ambiguous, costly, destructive, irreversible, or conflicts with these guardrails, **stop that action and report a proposed safe alternative**. This file does not grant additional tool permissions.

## 1. Required preflight for every autonomous session

Before editing anything:

1. Read this `AGENTS.md` from the current `main` branch; confirm its version and any changes since the previous session.
2. Read the current root `README.md`, the relevant source code, and the **live** [research-tracking Google Sheet](https://docs.google.com/spreadsheets/d/1le9GUaezNO4KWGSYG__CJDPCaszxJ7c8PzinLgGXFLc/edit).
3. Read the review evidence in `Review comments for the submission.pdf` (or an already verified transcription) before interpreting reviewer concerns.
4. Check the latest GitHub commit, existing modifications, related task IDs, and prior evidence. **Do not duplicate already-completed work.**
5. Choose **one narrowly scoped next action** that is authorized, testable, and realistically completable in the current run. Identify its source files, expected output, and verification procedure.
6. If any required source or tool is unavailable, mark the limitation **Blocked/Chưa thể kiểm chứng**; do not guess or simulate a successful run.

Do not trust stale summaries over current repository files or the live Sheet. If they conflict, disclose the conflict and verify its source.

## 2. Allowed unattended work

Permitted without additional approval, subject to access and verification:

- Read and audit Python scripts/notebooks, `manuscript/main.tex`, configuration files, dataset schemas, and experiment logs.
- Trace the current SERF pipeline and produce **evidence-backed** descriptions of routing, training, evaluation, and data flow.
- Create isolated, minimal, clearly named **CPU-friendly** audit scripts, unit tests, reproducibility checks, and experiment scaffolding.
- Prepare routing ablation implementations for confidence-only, margin-only, risky-class-only, full SERF, no-risk-class, and matched-rate random controls; **label all unexecuted experiments as unexecuted**.
- Run bounded local checks that require no paid service, no GPU training, and no large dataset or model downloads.
- Commit small, reviewed, reversible documentation/test/code changes to GitHub when the exact diff is understood and verified.
- Record verified results, evidence URLs/commit SHAs, and blocker reasons directly in the Google Sheet when accessible.
- Explain uncertainties, negative findings, and hypotheses still awaiting tests.

**Limit unattended scope:** Default to one coherent unit of work per scheduled invocation. Never use a periodic session as authorization for an unlimited implementation project.

## 3. Forbidden unattended actions (require fresh, explicit owner approval)

Do **not**:

1. Start or schedule expensive GPU training, Colab paid runtimes, cloud deployments, large-scale model downloads, external paid API calls, or other activities that may incur charges.
2. Delete, overwrite, rename, or replace original datasets, notebook training histories, saved checkpoints, logs, model outputs, the review PDF, or the rejected manuscript's source-of-record.
3. Rewrite experimental tables, abstract claims, reported scores, figures, citations, paper authorship, affiliation, or bibliography to appear better supported than existing evidence. **No fabricated numbers, experiments, citations, confidence intervals, or tests.**
4. Submit/resubmit a manuscript, email a reviewer or conference, alter repository visibility, permissions, secrets, branch protection, or third-party sharing.
5. Merge broad architectural refactors, change model/dataset labels silently, modify experiment protocol after seeing test performance, or switch research topics without an explicit owner decision.
6. Create parallel Markdown task/status trackers, duplicated Sheets, GitHub Issues, or task-management services to replace the canonical Sheet.
7. Install or execute untrusted dependencies or notebooks without understanding their effects; execute downloaded shell code, `curl | bash`, or arbitrary remote scripts.
8. Read unrelated private email, files, or connected-app content for this task; disclose credentials, personal data, or unpublished data to third parties.
9. Bypass failures by deleting tests, relaxing assertions, suppressing errors, changing labels, or omitting contrary results.
10. Claim a commit, test run, experiment, Sheet update, literature verification, or successful compilation happened unless the tool result confirms it.

Do not circumvent these restrictions indirectly (e.g., create a GitHub Actions workflow that silently launches the prohibited workload).

## 4. Scientific invariants (non-negotiable)

### 4.1 Freeze the legacy evidence; separate proposals from measurements

- The prior paper is a **rejected submission**, not a validated benchmark winner.
- Historical descriptive values include SERF Macro-F1 `0.7770 ± 0.0307` and static ELECTRA–RoBERTa Macro-F1 `0.7554 ± 0.0577` (three-seed summaries on a 146-instance fixed test set).
- These values are **not** evidence of statistically significant improvement. Do not describe the 2.16-point difference as proven superiority.
- Keep the original run artifacts untouched. Place new outputs under distinct paths with provenance; never overwrite legacy predictions with reruns.
- Write `HISTORICAL`, `PLANNED`, `IMPLEMENTED_NOT_RUN`, `RUN_VERIFIED`, or `UNVERIFIED` wherever status ambiguity is possible.

### 4.2 Avoid leakage and unfair comparisons

- Data splits must precede augmentation. Apply augmentation and class weighting using training data only.
- Select hyperparameters, thresholds, risky-class sets, and checkpoints on validation data, **not** the final test set.
- Use consistent datasets, compatible splits, training/augmentation conditions, seeds, evaluation metrics, and compute measurement setup when claiming cross-model comparisons.
- For cross-project evaluation, ensure no project IDs leak between train and held-out partitions.
- Save dataset versions, split identifiers, seeds, preprocessing, package versions, model IDs, and per-instance predictions.
- With only two to five minority-class test examples, interpret class-level F1 and aggregate Macro-F1 cautiously. Report uncertainty and support.
- Choose paired statistical methodology based on the actual sampling unit/dependence; do not run a generic test merely to obtain a p-value.

### 4.3 Falsify novelty rather than assume it

- The implemented router is `(confidence < tau) OR (margin < delta) OR (predicted class in R)`.
- Legacy risky set `R={A,L,FT,MN,PO,SC,PE,O,US}` contains **9 of 12 classes**.
- Legacy grid `delta ∈ {0.15, 0.20}` produced identical aggregate results; investigate whether its effect is zero.
- Test each component and report route rates, overlap among triggers, per-route correctness, matched baselines, and uncertainty.
- Include a class-only and a matched-rate random routing baseline; **do not rename class-only switching as novel confidence-aware routing**.
- If ablation fails to establish independent routing value, record a **PIVOT or STOP recommendation** rather than engineering the metrics to fit the hypothesis.

### 4.4 Efficiency is not encoder invocation count

- Legacy implementation independently trains the selector ELECTRA and the ELECTRA in the fusion branch, so the hard path can invoke three encoders.
- `1 + 2r` and the 13.24% legacy reduction are **encoder-invocation proxies**, not measured latency, memory, FLOPs, throughput, or energy.
- Reusing ELECTRA representations may require retraining or a different architecture; never claim equivalent accuracy before a controlled comparison.
- Compare against ELECTRA-only, static fusion, and lightweight baselines using measured costs under controlled hardware/batching when available.

## 5. Repository modification policy

- Prefer new narrowly scoped scripts/tests or incremental changes to current modules. Do not rewrite training notebooks wholesale during unattended runs.
- Before writing, inspect neighboring code, dependencies and git history for related changes.
- For each edit, identify: **why, files changed, risks, verification, rollback/commit**.
- Preserve interfaces and default behavior until a test demonstrates the alternative.
- Use small commits with informative messages such as `test(serf): cover router trigger combinations`; do not force-push, squash unrelated work, or rewrite history.
- Do not edit the Google Sheet or `README.md` merely to signal that a session ran.
- Never edit this `AGENTS.md` in an autonomous scheduled session. Only the project owner may authorize changes to these constraints.
- If a test requires a resource absent from the environment, report **NOT RUN**, not **PASS**.

## 6. Canonical tracking and status semantics

The **only live task tracker** is the Google Sheet:
`https://docs.google.com/spreadsheets/d/1le9GUaezNO4KWGSYG__CJDPCaszxJ7c8PzinLgGXFLc/edit`

- The `Theo dõi dự án` tab contains the 28-task plan for **one owner: Quách Thế Dương**.
- Keep `Dashboard`, `Reviewer_Responses`, `Milestone_Roadmap`, `Experiment_Metrics`, `RQ_Specification`, `Benchmark_Parameters`, and `Owner_Protocol` consistent with verified research.
- Do **not** re-create `tracking/README.md` or any alternate Markdown status checklist.
- Do not mark a task **Hoàn thành** unless its acceptance criterion is actually satisfied and its code/result/commit evidence is accessible.
- `Đang tiến hành` means a verifiable partial artifact exists, not merely that the AI planned to work on it.
- Leave dates/metrics blank when unknown; do not invent due dates, GPU specifications, confidence intervals, or runs.
- If writing the Sheet fails, report the failure, link the relevant commit, and leave the status unchanged. Do not quietly fall back to an alternate tracker.
- A completed code scaffold is **not** a completed experiment; distinguish `implementation-ready` from `experiment-run`.

## 7. Per-session execution protocol

Follow this order every time:

1. **READ**: `AGENTS.md` → current repo code and review evidence → live Sheet status → relevant previous commits.
2. **SELECT**: choose the next unblocked highest-priority task, preferably around router audit, routing ablations, matched experimental protocol, or reproducibility tests.
3. **PLAN**: state the smallest deliverable, expected files, acceptance check, and what must remain unchanged.
4. **EXECUTE**: inspect/edit only necessary files; avoid modifying raw data or running GPU training.
5. **VERIFY**: run available low-cost tests/static checks; distinguish verified success from unexecuted/inaccessible tests.
6. **COMMIT**: only if the change is correct, safe, small, and backed by checks. Record the SHA.
7. **TRACK**: update the exact relevant row(s) in the live Google Sheet with actual evidence and no speculative completion.
8. **REPORT** in Vietnamese: `task ID | finding/action | files and commit | checks PASS/FAIL/NOT RUN | Sheet status | blockers | next safe task`.

If blocked, stop without inventing output. Do not call a blocked or unverified task finished.

## 8. GO / PIVOT / STOP research gate

- **GO**: evidence establishes an independently justified routing contribution, credible matched comparative evaluation, and a meaningful predictive-quality versus real compute-cost trade-off; limitations remain explicit.
- **PIVOT**: class-only or cheap alternatives explain the gains, robust improvement is unconfirmed, or efficiency is not justified. Propose a narrower/new research hypothesis **without automatically changing the paper's research direction**.
- **STOP**: critical data/provenance is unavailable, methodology cannot be repaired within reasonable constraints, or further work requires owner approval/cost.
- A CORE B venue target is an **aspiration**, not a guarantee. Never claim publication readiness from checklist completion alone.

## 9. Escalation rules

Request an explicit owner decision **before** paid training, model replacement, dataset substitution, revising scientific claims, renaming/retiring SERF, publication, permissions changes, destructive work, or a significant new research direction.

**No owner response during overnight runs = no approval.** Continue only independent, low-risk tasks that satisfy these constraints; otherwise stop safely.

---

**Effective use:** This file is intended to be read explicitly at the start of every scheduled AI session. It complements—not overrides—system instructions and access controls. If tool access is unavailable, report the limitation; do not imply continuous execution or guaranteed completion.
