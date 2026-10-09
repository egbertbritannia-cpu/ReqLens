# SERF — Solo Project Tracking

**Owner:** Quach The Duong (Dương) — single contributor for planning purposes.  
**Project:** [rbl-requirements-classification](https://github.com/egbertbritannia-cpu/rbl-requirements-classification)  
**Review source:** [Review comments for the submission.pdf](../Review%20comments%20for%20the%20submission.pdf)  
**Paper:** [manuscript/main.tex](../manuscript/main.tex)  
**Current stage:** Recovery after EAI FISAT 2026 rejection (decision received 2026-10-04).  
**Schedule:** W1–W8 are *relative sprints*, not committed calendar dates.

## Progress

- Done: **1 / 28** (review intake/mapping only)
- In progress: **0 / 28**
- Not started: **27 / 28**
- Other task statuses: `In Progress`, `Needs Review`, `Blocked`, `Done`.
- A task is *Done* only after the linked deliverable and its acceptance evidence have been checked.

## Checklist (one owner only)

| ID | Sprint | Task | Owner | Status |
|---|---|---|---|---|
| P01 | W1 | Archive decision and reviewer action matrix | Dương | Done |
| P02 | W1 | Freeze submitted manuscript, runs and environment | Dương | Not Started |
| P03 | W1 | Audit selector/fusion code path | Dương | Not Started |
| P04 | W1 | Define falsifiable novelty and pivot criteria | Dương | Not Started |
| P05 | W1 | Freeze matched experimental protocol | Dương | Not Started |
| P06 | W2 | Ablate confidence-only routing | Dương | Not Started |
| P07 | W2 | Ablate margin-only routing | Dương | Not Started |
| P08 | W2 | Ablate risky-class-only routing | Dương | Not Started |
| P09 | W2 | Test full, no-class, matched-random routing | Dương | Not Started |
| P10 | W3 | Analyze per-instance easy/hard route errors | Dương | Not Started |
| P11 | W3 | Assess/retrain representation-reuse architecture | Dương | Not Started |
| P12 | W3 | Benchmark latency, throughput, peak memory and FLOPs | Dương | Not Started |
| P13 | W4 | Re-run single-encoder baselines without augmentation | Dương | Not Started |
| P14 | W4 | Re-run static hybrid baselines with matching protocol | Dương | Not Started |
| P15 | W4 | Add DistilBERT and TF-IDF+SVM cost controls | Dương | Not Started |
| P16 | W5 | Repeated validation with class support checks | Dương | Not Started |
| P17 | W5 | Paired CIs, statistical and sensitivity analysis | Dương | Not Started |
| P18 | W5 | Project-disjoint evaluation | Dương | Not Started |
| P19 | W6 | Audit modern external dataset provenance/labels | Dương | Not Started |
| P20 | W6 | External transfer test if admissible data exists | Dương | Not Started |
| P21 | W6 | Qualitative mixed FR/NFR error analysis | Dương | Not Started |
| P22 | W6 | Reproducibility manifest and scripts | Dương | Not Started |
| P23 | W7 | GO / PIVOT / STOP decision on novelty | Dương | Not Started |
| P24 | W7 | Rewrite RQs, claims, abstract and conclusion | Dương | Not Started |
| P25 | W7 | Trim manuscript to next venue's page limit | Dương | Not Started |
| P26 | W8 | Reviewer-response evidence traceability | Dương | Not Started |
| P27 | W8 | Clean compile, citations and reproduction QA | Dương | Not Started |
| P28 | W8 | Choose and validate next venue fit | Dương | Not Started |

## Review-to-experiment map

| Reviewer concern | Tasks | Required evidence |
|---|---|---|
| Router novelty, 9/12 risky classes, margin no effect | P03–P04, P06–P09, P23 | Component-wise ablations, route counts and matched-random control |
| Small test set and unsubstantiated performance margins | P05, P16–P18 | Repeated evaluations, paired CIs, class support |
| Augmentation and baseline confounds | P05, P13–P15 | Matched protocol and honest lightweight baseline comparison |
| Recomputing ELECTRA; performance-to-cost | P11–P12, P15 | Reuse study and measured latency, throughput and memory |
| No industrial/generalization or deep error analysis | P10, P18–P21 | Project-disjoint evaluation, dataset audit, qualitative failures |
| Incomplete reproducibility and excess pages | P02, P22, P24–P28 | Run manifests, citations/build checks and venue-compliant text |

## Stage gates

1. **G0 (W1):** Freeze scientific hypothesis, protocol and legacy baseline.
2. **G1 (W2–W3):** Demonstrate or falsify independent contributions of the router components.
3. **G2 (W3–W4):** Matched baseline and measured computation trade-off.
4. **G3 (W5):** Paired uncertainty and cross-project checks.
5. **G4 (W6):** External/generalization evidence where feasible; explicit limitations otherwise.
6. **G5 (W7):** Document **GO / PIVOT / STOP** decision; do not assume novelty.
7. **G6 (W7–W8):** Release candidate with reproducibility, page, reference and claim audit.

## Operating rules

- **Single owner:** Dương handles all tasks. Do not imply independent reviewers, double screening or delegated checks.
- **Test-set discipline:** Tune on validation only. Preserve run IDs, seeds, code commits, grouping and label distributions.
- **Reproducibility:** Record every figure/table's provenance and every failure or unavailable artifact explicitly.
- **Truthfulness:** The prior reported SERF Macro-F1 0.7770 ± 0.0307 and static-fusion 0.7554 ± 0.0577 are descriptive, not proof of superiority.
- **Cost:** 1.7352 active encoder calls is a *proxy*, not measured latency or memory benefit. Selector and fusion encoders were independently trained, so reuse may require retraining.
- **Update discipline:** When a task changes status, update both this checklist and the corresponding workbook `Project_Plan` tab. The workbook is a separately delivered copy; it is **not automatically synchronized** with GitHub.

## Workbook layout

A companion eight-tab spreadsheet (`SERF_Solo_Project_Tracking.xlsx`) contains `Dashboard`, `Project_Plan`, `Owner_Protocol`, `Benchmark_Parameters`, `RQ_Specification`, `Reviewer_Responses`, `Milestone_Roadmap`, and `Experiment_Metrics`. It is derived from the supplied `project-tracking.xlsx` layout but rewritten for the SERF solo recovery project, not the former CodeNet/Randoop team project. Do not present spreadsheet results for unexecuted experiments as measured outcomes.
