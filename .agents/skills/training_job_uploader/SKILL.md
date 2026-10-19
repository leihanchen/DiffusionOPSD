---
name: training_job_uploader
description: Find and sync a Slurm job's local W&B runs, summarize verified training and evaluation metrics, and post a new comment with online run links to a specified Linear issue. Diagnose missing W&B logs when no matching run is found.
---

# Training Job Uploader

## Invocation and scope

Accept two positional arguments:

```text
$training_job_uploader JOB_ID LINEAR_ISSUE_ID
$training_job_uploader 6166613 EAS-8
```

The first argument identifies the Slurm job; the second identifies the Linear issue receiving a **new, separate comment**. Ask for a missing argument. Work from this project's root, deriving paths from the checkout rather than hardcoding a cluster or home directory.

Invocation authorizes syncing the matching W&B run(s) and posting the requested summary. Missing-log debugging means investigation and diagnosis; it does not authorize code changes, commits, job cancellation, or resubmission. Do not invoke the broader repair-and-resubmit workflow automatically.

## 1. Locate the job's W&B records

- Recover job state, exit code, working directory, stdout/stderr, and effective run directory from `scontrol show job JOB_ID`, `sacct`, submission records, and job logs. Completed jobs may no longer appear in `scontrol`. Resolve array/step identifiers to the actual requested scope.
- Search the identified run directory and project `logs/` and `wandb/` locations for `offline-run-*`, `run-*.wandb`, W&B configuration/debug logs, manifests, and `metrics.jsonl`. Use `rg --files --hidden --no-ignore` restricted to these directories when normal file discovery excludes ignored logs. Follow job-specific paths outside the checkout when recorded by the job.
- Establish the association using the job's stdout W&B path/run ID, manifests, or saved job metadata. A matching timestamp or the `latest-run` symlink alone is insufficient. Do not upload unrelated runs. If multiple runs belong to the job, account for each; distinguish resumed history from updates produced by this job.
- If no matching local W&B run is found, follow **Missing-log diagnosis** below. An inaccessible filesystem or unavailable cluster is an unverified search, not proof that logs do not exist.

## Run naming

Every uploaded W&B run's **display name** must contain the job ID, a concise job purpose, and the cluster where the job executed. Use:

```text
job-JOB_ID-PURPOSE-CLUSTER
job-6186524-base-step5-step7-eval-killarney
job-6166613-wan22-opsd-eight-worker-probe-killarney
```

- Derive the purpose from the actual workload, submission script/configuration, and job logs; use a readable lowercase hyphenated phrase. Recover the execution cluster from scheduler accounting, saved submission metadata, or job logs. The uploader's current hostname is not evidence of the source cluster when logs were copied.
- If the purpose or cluster cannot be established, ask the user for the missing value before syncing or posting. Do not use invented labels or omit required fields.
- For multiple runs from one job, append a phase label or the original run ID to distinguish them. Preserve the underlying W&B run ID and URL; the display name is a separate field.
- Invocation authorizes setting this display name on the exact matched online run. After sync, use the W&B API to update the existing run's display name (`run.name = display_name; run.update()`), then fetch it again and verify the persisted name before posting. Apply the rule to already-synced runs too; do not create another run to rename one. Do not rewrite the offline log files or rename their directories.
- If naming fails, report that the upload succeeded but naming is blocked; do not claim the naming requirement is complete or post a completed-workflow summary.

## 2. Resolve destination and sync

- Resolve the exact W&B **entity/account and project** from explicit user instructions, established project/session settings, or the matched run's saved configuration. Do not assume the authenticated user's personal workspace is the destination. Preserve underscores/hyphens exactly; do not silently substitute similarly named entities.
- If either destination field is missing or conflicting, ask the user for the missing/ambiguous value before uploading. Continue local analysis while awaiting the answer. Do not hardcode `cvis_tmu/diffusionopsd` as a universal default.
- Use existing W&B authentication. If the user identifies `~/.bashrc` as the credential source, load only the relevant credential into the subprocess environment without printing it, dumping the file, or sourcing unrelated shell commands. Never put secrets in command arguments, reports, or tracked files. Ask the user to authenticate when no usable credentials are available.
- Sync the exact run directory with explicit destination arguments, preserving the original run ID:

```bash
wandb sync --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$RUN_DIRECTORY"
```

- Run the sync process in online mode without changing the training job's offline environment or configuration. Honor an explicitly configured W&B server. Preserve local records and media. Do not create a replacement run merely because the original is already synced.
- For an active job, identify the upload as a partial snapshot; do not claim final completeness. Do not stop training to obtain a final log.
- Verify the exact remote run using the W&B API: destination, run ID, required display name, online URL, recorded steps, key metric values, and expected media files/counts. Compare against local records. A successful CLI exit alone is insufficient evidence of complete history/media. Report any omissions; a failed job can still have useful partial logs.
- On transient failures, retry at most twice after inspecting the cause. Stop on persistent authentication, permission, or destination errors and report the blocker. Do not delete sync markers or use broad `--sync-all` recovery.

## 3. Summarize measured performance

Use verified online history plus local job evidence, separating their sources where they differ. Include available information, and label missing metrics rather than inventing values:

- Job state/exit code, cluster, allocation, elapsed time, checkpoint provenance, completed updates, and checkpoint/output locations.
- Training losses, gradient norms, rollout and identity-retained counts, geometry/reward scores, effective weights, motion/quality masking, and consistency checks.
- Held-out evaluation values and deltas, baseline identity, prompt/seed counts, and evaluation steps. Distinguish training batches from matched evaluation comparisons.
- Update and phase timings, GPU allocated/reserved peaks, host memory, and allocated GPU-hours when the allocation/time are known. Do not sum overlapping worker phase maxima.
- Improvements, regressions, failure warnings, uncertainty, and a proportionate next action.

Interpret metric definitions from the producing code/configuration. In this project's current OPSD implementation, higher `frac_motion_masked`/`frac_quality_masked` means more samples fell below the corresponding threshold. Different prompts/seeds across updates prevent attributing this alone to model regression. `mean_w = 0` does not mean no optimization: verify the branch loss, which can still train through the negative branch. Loss reduction and tiny held-out samples do not establish broad generation-quality improvement.

## 4. Post a new Linear comment

- Resolve the exact requested issue using the available Linear connector; confirm its identifier/title. Do not infer a different issue from previous conversations.
- After syncing, naming, and verifying the available records, create one new top-level comment whose heading contains the job ID, job purpose, and execution cluster. Include the verified W&B display name(s), concise performance summary, limitations/completeness, and each verified online run URL. Include relevant local artifact paths as recovery evidence. Clearly label partial uploads.
- Use the connector's create/save-comment operation with the issue ID and Markdown body. Omit existing comment ID and parent ID so this is a separate comment, not an edit or reply. Do not modify the issue description, status, or checklists.
- If the write response is uncertain, read the issue's comments for the same job/run and summary before retrying to avoid duplicates. A fresh explicit invocation may request a new comment; a transport retry must not create another copy.
- If Linear is unavailable, retain the prepared summary in an ignored local logs directory and return its path plus the verified W&B URL. Do not claim it was posted. If W&B sync is blocked, report that blocker rather than posting a completed-upload summary.

## Missing-log diagnosis

State: **“No local W&B logs found for job JOB_ID in the searched locations.”** List those locations and any access limits. Then investigate:

1. Check scheduler state and the first actionable error in stdout/stderr. Pending jobs or failures before `wandb.init` may never produce a run.
2. Trace the job's actual launch configuration, `WANDB_MODE`, `WANDB_DIR`, logger enablement, rank-zero initialization, container bind paths, and working directory. Check whether logging was disabled, redirected, online-only, or interrupted before initialization.
3. Inspect local metrics/manifests and W&B debug logs, if present, to distinguish “training never started” from “training ran but logging failed.” If a verified online-only run exists, report its URL and explain that no offline sync is possible; do not call it an uploaded local run.
4. Report the evidence-backed cause or unresolved hypotheses, the last completed training stage, and the smallest next diagnostic/fix. Do not launch GPU workloads on a login node or manufacture a W&B run from text logs without a separate request.

No-log diagnosis does not create a success comment in Linear. Return the findings to the user; a failure-investigation comment requires a separate request.

## Completion report

Return the job ID, purpose, cluster, sync state (complete/partial/blocked/not found), verified W&B display name(s) and URL(s), and Linear comment confirmation/link. Highlight material quality concerns or missing evidence. Do not equate a finished W&B run with a successful Slurm job.
