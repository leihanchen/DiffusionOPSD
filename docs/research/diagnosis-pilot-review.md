# Reviewing the 12-clip diagnosis pilot

This pilot compares Gemma with a human reviewer on 12 existing Wan2.2 training-source
clips. It checks for obvious diagnosis mistakes before using observations to edit
prompts. It is not a precise population accuracy estimate, a physics benchmark, or
proof that these 12 clips cover all future model failures. Several clips share scene
families (for example chessboards); report that limitation rather than treating 48
judgments as 48 independent videos.

## First: save independent human labels

The prepared packet is under ignored `logs/curriculum-preflight/review12/`. Open
`review.html` in a browser, or download/extract `review12.zip` and open the same file.
It contains prompts, 96 selected frames, videos, and blank fields. It deliberately
contains no Gemma predictions. A CSV alternative is `labels_blank.csv`.

Enter a reviewer name or ID. For each clip:

1. Check the prompt belongs to G1: stationary rigid scene/object with camera motion.
   Mark any unsuitable case in your notes and replace it before treating the set as
   a complete 12-clip audit. Do not silently change the prompt to fit the output.
2. Inspect the eight ordered frames and their source frame IDs/timestamps. These
   match the information supplied to Gemma. Full video playback is supplemental;
   record events only visible between sampled frames separately, not as failures
   Gemma could have detected from its inputs.
3. Assign one label per requirement using the rubric below. For pass/fail, cite at
   least one displayed source frame ID. Usually cite two views for a temporal claim.
   Give a short visible reason; unknown also needs an explanation.
4. Export `reviewer_labels.json` before seeing Gemma answers. Keep that original
   file unchanged during comparison; separately record any corrected annotation.

| Requirement | observed_pass | observed_fail | unknown |
| --- | --- | --- | --- |
| Camera motion | Visible camera movement matches the requested path | Visible output contradicts the requested motion, e.g. a clearly static view instead of orbit | Camera and object motion cannot be separated, or sampled frames do not reveal the path |
| Stationarity | Objects stay fixed relative to the scene while the camera may move | An object visibly drifts, rotates, or deforms independently of the intended static scene | Reference/background evidence is insufficient to distinguish movement |
| Identity | The same object retains its identifying appearance | Object identity switches, disappears, duplicates, or changes identifying features | Occlusion or lack of visible detail prevents a judgment |
| Geometry | Observable shapes and spatial relations remain consistent across available views | Visible shape/structure warps, disconnects, or changes inconsistently | Views do not expose the structure; no hidden 3D accuracy claim is possible |

Example annotation (illustrative, not a label for an actual candidate):

```json
{
  "clip_id": "legacy_126",
  "request_hash": "copied automatically from the frozen packet",
  "reviewer": "reviewer-name",
  "labels": {
    "camera_motion": "observed_fail",
    "stationarity": "observed_pass",
    "identity": "observed_pass",
    "geometry": "unknown"
  },
  "human_evidence": {
    "camera_motion": {"frame_ids": [0, 16], "reason": "Viewpoint remains static despite requested orbit."},
    "stationarity": {"frame_ids": [0, 16], "reason": "Object stays fixed relative to its pedestal."},
    "identity": {"frame_ids": [0, 16], "reason": "Identifying features remain unchanged."},
    "geometry": {"frame_ids": [], "reason": "Only one side is visible."}
  }
}
```

## Then: produce the comparison

Run Gemma diagnosis using the prepared ordered requests, the same pinned model,
processor, template, eight-frame selection, and decoding limits as the audit. The
packet preparation saves requests outside the reviewer packet at
`logs/curriculum-preflight/review12_gemma_requests.json`. Do not show predictions
while the human is labeling. Each answer must pass the normal evidence/schema gate. The page exports each
clip request hash with its human labels, binding them to the prompt and frame bytes.
Regenerated inputs cannot silently reuse older labels.
The comparison tool accepts only matching model identities, clip counts, IDs, and
valid human evidence; request hashes also bind answers to inputs when recorded.

```bash
bash scripts/run_cosmos_python.sh scripts/audit_curriculum_diagnosis.py \
  --manifest logs/curriculum-preflight/review12/manifest.json \
  --labels /absolute/path/reviewer_labels.json \
  --predictions /absolute/path/gemma_responses.json \
  --output-dir logs/curriculum-preflight/review12-agreement
```

The outputs are `agreement_report.json`, `agreement_summary.csv`, and
`disagreements.json`. The report is always initially unapproved. The default audit
size is now 12; 24 remains available by explicitly selecting `expected_count=24`.
No human labels have been produced by preparing this packet.

## How to read the agreement report

Read each requirement separately. A combined percentage can hide a weak camera
motion or geometry evaluator. With 12 clips, one disagreement changes a requirement's
agreement by 8.3 percentage points; report counts alongside percentages.

| Field | Meaning | What to inspect |
| --- | --- | --- |
| agreement_rate / disagreements | Exact human–Gemma match, including unknown | Every disagreement, with both reasons and frame references |
| false_pass | Human says fail; Gemma says pass | Missed visible failures that could keep bad tasks in the curriculum |
| false_fail | Human says pass; Gemma says fail | Invented failures that could cause unnecessary or misleading prompt edits |
| overconfident_on_unknown | Human says unknown; Gemma makes a definite judgment | Unsupported claims, especially hidden geometry or ambiguous camera/object motion |
| unknown_on_decidable | Human gives a definite label; Gemma abstains | Whether diagnosis provides enough usable feedback |
| unknown_rate | Fraction of Gemma answers marked unknown | Interpret with the human class distribution; abstention is sometimes correct |
| missing_human_classes | Pass/fail/unknown classes absent from this requirement's sample | Gaps in what the pilot can validate; do not invent labels to fill gaps |

For example, 11/12 matches does not establish useful failure detection if all
12 human examples are passes. If geometry has only unknown labels, the pilot tests
appropriate abstention, not geometry error detection. The report records coverage
gaps instead of refusing to produce a comparison.

Review `disagreements.json` alongside the displayed frames. Decide whether each
mismatch is a Gemma error, ambiguous rubric, sampling limitation, or human annotation
error. Save the explanation separately. If the rubric/template changes, rerun the
pilot and identify the new version; do not tune labels to agree with Gemma.

Only a human reviewer can approve use after inspecting the report. There is no
scientifically calibrated automatic percentage threshold for this 12-clip pilot.
Record the decision, reviewer, observed failure patterns, and limitations. If there
are repeated false failures or unsupported confident judgments, keep the numeric-only
curriculum, revise the diagnosis rubric, and repeat the review. Do not claim a
requirement is validated when its relevant failure classes are absent.

For an explicit approval, set `approved: true` and `approved_by` on the unchanged
report. The report preserves the audited input/request and diagnosis template identities.
Changing the active diagnosis rubric invalidates approval. The existing report
hash protects against accidental changes to the measured results; it is not an authentication mechanism or a substitute for genuine review.
Approval is a pilot decision, not certification of physical accuracy.
