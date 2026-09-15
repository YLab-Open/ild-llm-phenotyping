#  Zero-shot LLM phenotyping of drug-induced interstitial lung disease

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.19.0-FF6B6B.svg?style=flat)](https://github.com/vllm-project/vllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### Identifying interstitial lung disease with large language models on clinical notes and claims

Larghero G, Gu B, Anand P, Lin KJ, Schneeweiss S+, Yang J+.

*+Correspondence to Sebastian Schneeweiss and Jie Yang.*

#### ABSTRACT

Drug-induced interstitial lung disease (ILD) is a rare but sometimes fatal toxicity of immune checkpoint inhibitors and antibody-drug conjugates. Post-marketing surveillance depends on finding these cases in large healthcare databases. Manual chart review is accurate but does not scale, so surveillance relies on rule-based algorithms applied to diagnosis codes; these must be hand-built by experts, and they trade the purity of the case series against the number of cases identified. We assessed whether large language model (LLM) phenotyping improves on published claims-based algorithms, and whether any gain yields a measurable epidemiological benefit.

In a retrospective study of 309 oncology patients with ILD diagnosis codes at Mass General Brigham, we compared 12 open-weight LLMs with 20 published claims-based algorithms against expert adjudication. Probable and possible ILD were classified as positive. LLMs received claims, notes, or both. Outcomes were positive predictive value (PPV), yield (proportion classified positive), and projected association attenuation and sample-size requirements.

The LLMs extended the PPV--yield frontier of the published algorithms under all three inputs, and did so without any task-specific training. Averaged across the 36 model--input evaluations, PPV exceeded that of the most inclusive rule by 19·7 percentage points. With claims alone, Qwen3-8B achieved 89·8\% PPV (95\% CI 78·2--95·6) at 15·9\% yield, versus 86·7\% (62·1--96·3) at 4·9\% for the highest-PPV rule: a higher PPV while classifying three times as many patients. With notes alone, Qwen3-32B achieved 78·9\% PPV (71·8--84·7) at 49·2\% yield, versus 64·6\% (56·8--71·6) at 51·1\% for the strongest non-specialist rule, a 14·3-point gain at comparable yield. With both inputs, Qwen3-14B achieved 100·0\% PPV (95·1--100·0) at 24·3\% yield. Under non-differential misclassification (true odds ratio 3·0; baseline ILD prevalence 10\%), the projected observed odds ratio was 2·67 for Qwen3-14B with both inputs, versus 1·07 for the strongest non-specialist rule. Detecting that association at 80\% power would require 591 patients with LLM phenotyping, against approximately 27,200 with the rule.

Zero-shot LLMs offer a higher-PPV alternative to rule-based claims phenotyping for drug-induced ILD, whether the input is structured claims, clinical notes, or both. Because the claims-only configuration uses exactly the codes that existing algorithms consume, it is an immediately deployable upgrade for administrative databases without linked notes, and may recover drug-safety signal otherwise lost to outcome misclassification.

<p align="center">
  <img src="docs/fig1.png" width="900" alt="Study design: cohort construction, rule-based comparators, and the LLM phenotyping pipeline">
</p>

Code accompanying a study that used open-weight large language models to
identify interstitial lung disease (ILD) in cancer patients from clinical
notes and administrative claims, and compared them against 20 published
claims-based algorithms on a 309-patient adjudicated cohort. Thirteen models
were run through a single deterministic pipeline across three input
configurations (claims only, notes only, fused); twelve are in the reported
set.

> Citation: [PLACEHOLDER — paper citation to be added on publication]

## Data availability

The input data are protected health information and **cannot be distributed**.
No data are included in this repository and none of the pipeline can be run
as-is by a third party. The code is provided for transparency: to document
exactly how the cohort was built, how prompts were serialized, how inference
was run, and how every reported number was produced.


## Pipeline

The pipeline is `s1` → `s2` → `s3` → `s4` → `s5`, gated by `v1`
(serialization), `v2` (inference) and `v3` (results):

1. `stages/s1_build_cohort.py` — build the 309-patient core cohort from the
   adjudicated 360.
2. `stages/s2_serialize.py --config {claims,notes,fused}` — render prompts;
   notes are paginated into batches costed under every model's tokenizer.
   Gate: `validate/v1_serialization.py` — every patient present, every batch
   fits every model's context, each note in exactly one batch, window bounds.
3. `stages/s3_infer.py --model M --config C` — the only GPU stage; vLLM with
   constrained JSON decoding, deterministic and resumable.
   Gate: `validate/v2_inference.py` — per cell: batch counts match
   serialization, no sentinel patients, labels in the closed set,
   probabilities valid, decoding parameters match `shared/config.py`.
4. `stages/s4_aggregate.py` — per-batch records → per-patient predictions.
5. `stages/s5_metrics.py` — per-cell metrics and the model × configuration
   grid (`results/grid_metrics.csv`).
   Gate: `validate/v3_results.py` — confusions sum to 309, gold labels match
   the reference standard, intervals bracket their points.

## Analysis scripts (`analysis/`)

All read `results/grid_metrics.csv` and the per-cell outputs; all CPU.

- `shen_evaluation.py` — the 20 published claims-based rules on the cohort.
- `frontier.py` — PPV–yield Pareto dominance of models vs the rule frontier.
- `power_projection.py` — Copeland misclassification / power model for every
  model × configuration and every rule.
- `decomposition.py` — claims/notes/fused modality decomposition and
  head-to-head McNemar tests against every rule.
- `aggregation_sweep.py` — alternative aggregation rules and probability /
  confidence threshold sweeps.
- `finegrained.py` — the note-tier ablation (plan, subcohorts, summary).
- `make_tables.py` — manuscript tables as LaTeX fragments.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `ILD_ROOT` | the repository checkout; overrides the auto-detected location | auto |
| `ILD_DATA_DIR` | input data directory, if not `shared/data/` | `shared/data/` |
| `ILD_MODEL_SHARE` | root directory of model checkpoints | `./models` |
| `ILD_DX_MAP`, `ILD_PX_MAP` | ICD / CPT mapping CSVs | under `ILD_DATA_DIR` |

To point the pipeline at your own data: set `ILD_ROOT` (or `ILD_DATA_DIR`) and
`ILD_MODEL_SHARE`, run `bash scaffold.sh`, then follow the pipeline order
above, running each gate before the next stage.

## Installation

```bash
conda env create -f environment.yaml
conda activate ild-llm
```

`environment.yaml` is the minimal environment: direct dependencies only, with
the versions actually used. `vllm` and `torch` are needed only for
`stages/s3_infer.py`; everything downstream runs on CPU.
