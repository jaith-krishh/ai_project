# Acoustic Sound Analyzer — Requirements

## Scope

This project has **exactly four requirements**. Nothing beyond this list should be built (no dashboards, no video pipeline, no Docker, no Grad-CAM, no FastAPI service — unless requested later).

Input: a single audio file.
Output: a report identifying the sounds in it.

---

## 1. Detect all sounds present in the audio

- Take an audio file as input.
- Detect **all** distinct sound events present, including overlapping ones (e.g. traffic + bird + construction happening at the same time).
- For each detected sound, output:
  - Sound label (e.g. "traffic", "bird chirp", "drilling")
  - Confidence score
  - Time range (start–end) where it occurs

**Approach:** Multi-label sound event detection model (CNN with sigmoid output, one node per known class) run over mel-spectrogram frames of the audio, with temporal post-processing to merge frame-level detections into event intervals.

---

## 2. Flag unknown sounds

- If a detected sound event does not match any known class with sufficient confidence, mark it as **"Unknown"** instead of forcing it into a known label.
- Report the time range of the unknown event just like known ones.

**Approach:** Extract an embedding vector for each detected event from the model. Compare it to per-class centroids (computed from training data). If the distance to every centroid exceeds a calibrated threshold, mark it as unknown.

---

## 3. Similarity list for unknown sounds

- For every sound marked "Unknown," output a ranked list of the **top 3–4 closest known sound classes**, with a similarity percentage for each.
  - Example: `Unknown sound @ 02:13–02:18 → 70% similar to "drilling", 20% similar to "engine idling", 10% similar to "jackhammer"`

**Approach:** Reuse the same embedding-to-centroid distances computed in requirement 2. Instead of discarding them once "unknown" is decided, sort by distance (closest first), convert to a normalized similarity percentage, and keep the top 3–4.

---

## 4. Classify the location type

- Based on the percentage breakdown of detected sound categories across the whole recording, classify the location as one of:
  - **Industrial** — dominated by traffic, construction, machinery-type sounds
  - **Natural habitat** — dominated by birds, wind, insects, water, other natural sounds
  - **Mixed / Residential** — no single category dominates

**Approach:** Simple rule-based classifier on top of the sound-category percentages already computed:
  - If (traffic + construction + machinery + engine) % > 60% → Industrial
  - If (bird + wind + insect + natural) % > 60% → Natural habitat
  - Otherwise → Mixed / Residential

(Exact category groupings and threshold can be tuned once real output percentages are seen.)

---

## Final Output Format (per analyzed file)

```
1. Detected sounds:
   - Traffic       | 00:00–00:45 | confidence: 0.91
   - Bird chirp    | 00:10–00:30 | confidence: 0.84
   - Unknown       | 02:13–02:18 | closest matches: 70% drilling, 20% engine idling, 10% jackhammer

2. Location classification: Industrial (traffic + construction = 68% of total audio)
```

---

## Explicitly Out of Scope

- Web dashboard / React frontend
- Video input / video-only pipeline
- Grad-CAM or model visualization
- Docker/containerization
- FastAPI or any web service layer
- Diversity index, "Good/Moderate/Poor" acoustic quality rating (unless later requested)

Only the four requirements above should be implemented.
