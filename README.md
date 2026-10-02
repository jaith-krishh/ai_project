# Acoustic Sound Analyzer

Takes one audio file and reports:

1. Every sound detected, with its time range and confidence (overlapping sounds included).
2. Sounds that don't match any known class, marked **Unknown**, with the 3 closest known classes and a similarity percentage for each.
3. The location type: **Industrial**, **Natural habitat** or **Mixed / Residential**.

See `requirements_new.md` for the full specification.

```
1. Detected sounds:
   - Traffic       | 00:00–00:45 | confidence: 0.91
   - Bird chirp    | 00:10–00:30 | confidence: 0.84
   - Unknown       | 02:13–02:18 | closest matches: 70% drilling, 20% engine idling, 10% jackhammer

2. Location classification: Industrial (traffic + construction = 68% of total audio)
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

The trained model is already in the repo at `outputs/checkpoints/best_model.pt` (60 classes from ESC-50 and UrbanSound8K, ~87% validation accuracy).

The datasets are only needed to train, compute centroids or calibrate the threshold, not to analyse a file. Place them under `data/`:

```
data/
├── ESC-50/
│   ├── meta/esc50.csv
│   └── audio/*.wav
└── UrbanSound8K/
    ├── metadata/UrbanSound8K.csv
    └── audio/fold1 … fold10/*.wav
```

## One-time preparation: centroids and unknown threshold

Unknown detection needs two files produced from the dataset. Run these once (from the project root) and commit the results:

**1. Class centroids** → `outputs/centroids.pt`

```bash
python -m backend.compute_centroids --data-dir data
```

Runs every training clip through the model and stores the average embedding of each class.

**2. Unknown threshold** → `outputs/threshold.pt`

Best option: a folder of real recordings of sounds that are *not* among the 60 classes:

```bash
python -m backend.calibrate_unknown --data-dir data --unknown-dir path/to/unknown_audio
```

Alternative: treat one known class as unknown. Its centroid (and any same-named class from the other dataset) is removed for the calibration. Pick a class that sounds unlike the others; avoid siren, car horn and dog, which appear in both datasets.

```bash
python -m backend.calibrate_unknown --data-dir data --holdout-class esc_church_bells --holdout-as-unknown
```

With neither option the script falls back to synthetic white noise, which is only good enough for testing.

## Analysing a file

```bash
python -m backend.run_analysis --input path/to/recording.wav
```

Any format librosa can read works (WAV, FLAC, OGG, MP3). If `centroids.pt` / `threshold.pt` are missing, the analysis still runs but prints a warning and skips Unknown detection.

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | `outputs/checkpoints/best_model.pt` | Trained model |
| `--centroids` | `outputs/centroids.pt` | Output of `compute_centroids` |
| `--threshold` | `outputs/threshold.pt` | Output of `calibrate_unknown` |
| `--window` | `4.0` | Window length in seconds (matches training clip length) |
| `--hop` | `2.0` | Step between windows in seconds |
| `--prob-threshold` | `0.3` | A window always reports its most likely class; other classes are also reported when their probability is at least this. Raise it if too many overlapping sounds appear, lower it if overlaps are missed. |
| `--top-k` | `3` | Closest known classes listed for each Unknown sound |
| `--json PATH` | – | Also save events and percentages as JSON |

From Python:

```python
from backend.run_analysis import analyze_file

result = analyze_file("recording.wav")
print(result["report"])
result["events"]       # [{"label", "start", "end", "confidence", "similar_to"}, ...]
result["aggregation"]  # label / category percentages and location
```

## How it works

| Step | Module | Owner |
|---|---|---|
| Mel-spectrogram + sliding window inference: 4 s windows every 2 s, softmax class probabilities and a 128-dim embedding per window | `backend/utils.py`, `run_analysis()` in `backend/run_analysis.py` | P2 Gautham |
| Class centroids and unknown-distance threshold | `backend/compute_centroids.py`, `backend/calibrate_unknown.py` | P1 Abhinav |
| Label windows (Unknown if the embedding is further than the threshold from every centroid), merge them into events, similarity list for Unknown events | `backend/p3_event_merging.py` | P3 Sarah |
| Percentage of audio per sound and category, location rule, text report | `backend/aggregate_report.py` | P4 Joe |
| Single `--input` command, README | `analyze_file()` / `main()` in `backend/run_analysis.py` | P5 Jaith |

Details:

- **Overlapping windows:** each window only counts the time up to the next window's start, so no second of audio is counted twice.
- **Overlapping sounds:** each sound is merged on its own timeline, so two simultaneous sounds produce two events.
- **Location rule:** a category's share is the union of time covered by its sounds. Industrial if industrial sounds cover more than 60% of the audio, Natural habitat if natural sounds cover more than 60%, otherwise Mixed / Residential. Household, human and pet sounds count towards neither. The class-to-category list is `CATEGORY_MAP` in `backend/aggregate_report.py`.

## Tests

```bash
python -m pytest tests
```

Tests that need the dataset or `outputs/centroids.pt` are skipped when those aren't present. `python tests/run_p3_demo.py` prints a worked P3 example.

## Retraining

```bash
python -m backend.train --data-dir data --epochs 30
```

After retraining, rerun both preparation steps; centroids and threshold are tied to a specific checkpoint.
