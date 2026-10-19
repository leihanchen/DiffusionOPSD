---
name: slurm_job_debugging
description: Monitor a Slurm job through completion, diagnose failures, fix and commit relevant code, and resubmit within a three-submission limit. Use when asked to run this recovery loop for a batch script path or job ID; a status-only request does not authorize repairs or submissions.
---

# Slurm Job Debugging

Accept a Slurm batch script path or an existing job ID. Check immediately, then every **10 minutes** while the job is active. Continue through diagnosis, repair, validation, commit, and resubmission until the workload succeeds or the submission budget is exhausted.

## Scope and submission budget

- Invoking this recovery workflow authorizes relevant fixes, local Git commits, and at most **three new submission attempts**. Do not push commits. Honor narrower user instructions, including requests to explain before editing.
- With a script path, its initial submission counts as attempt 1. With an existing job ID, that prior submission does not count; up to three new submissions remain. State this interpretation at startup.
- Count each actual `sbatch` invocation before executing it, including rejected submissions. `sbatch --test-only` is validation and does not count. Never submit attempt 4, reset the counter after a new error, or evade the limit with requeue, arrays, or another agent.
- Keep only one tracked attempt active at a time. After the third submission, continue monitoring that job to its terminal outcome. If it fails, explain the remaining error and stop without another repair-and-submit cycle.
- Preserve the intended workload and resource constraints. Do not silently shrink training, disable checks or rewards, change data/models, or declare success after a no-op. Resource changes must be justified by evidence and stay within the user's authorized scope.

## Initialize and preserve state

Resolve the cluster, repository, submission working directory, script, arguments, and expected success evidence. Read repository instructions and inspect Git status before edits. For a job ID, inspect `scontrol show job JOBID` and accounting records to recover `Command`, `WorkDir`, `SubmitLine`, `StdOut`, and `StdErr`. Completed jobs may have disappeared from `scontrol`; use `sacct` and local scripts/logs. Ask for missing script or cluster information only when it cannot be recovered. Never execute recovered command text blindly.

Create a small JSON ledger outside version control, under `${CODEX_HOME:-$HOME/.codex}/slurm_job_debugging/`, with a unique run filename. Record:

- original input, cluster, repository, script, submission directory and arguments;
- submission limit 3, attempts used, current job ID, next check time;
- each attempt's commit SHA, submission timestamp, job ID, status, exit code, and log paths;
- diagnosis, validation results, expected success evidence, and final outcome.

Update the ledger before a submission and immediately after receiving its ID. Resume the same ledger after interruption or context compaction. If interrupted between recording an attempt and receiving its ID, conservatively retain the consumed attempt until evidence proves no invocation occurred. If a submission response is lost, reconcile queue/accounting records before doing anything that could create a duplicate. Record the ledger path in progress updates.

For an existing active job, monitor it first. For a script input, check syntax, required submission paths, and relevant configuration, then submit from its intended directory. Ensure Slurm output directories exist **before** submission: `mkdir` inside the batch script is too late for Slurm to open its logs. Do not manufacture a commit if no repair was needed.

## Monitor

Use `squeue` for live state and `sacct` for final state, exit codes, and batch/step records. For example:

```bash
squeue -j JOBID -o '%.18i %.16T %.60R'
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList -P
```

An empty queue result is not success. Allow for accounting lag and query errors; unknown status must not trigger resubmission. Inspect relevant steps as well as the parent allocation. For arrays or multi-job workflows, establish the intended task scope before submitting anything that could repeat successful tasks.

While pending, running, configuring, suspended, requeued, or completing, schedule the next scheduler check for 600 seconds later. Do not spend another submission because the job is queued. Use interruptible waits; if an individual wait is limited to 60 seconds, split the wait while preserving the ten-minute scheduler cadence. Respond to user input during waits. Keep the interaction alive while monitoring; do not claim that a background service exists or that checks will continue after ending the session. If continued execution is unavailable, save the ledger and report that monitoring stopped.

## Diagnose and repair a failed attempt

1. Read final accounting, stdout, and stderr using the actual job's paths. Expand Slurm filename substitutions such as `%j`, `%x`, `%A`, and `%a`. Locate the first actionable error and follow its traceback or shell failure; distinguish warnings from the fatal error.
2. Trace the failure through the submitted script, effective configuration, container/environment, mounts, model/data paths, and application call site. Reproduce the smallest failing operation in the job's actual environment. Keep heavy model loading and GPU tests off login nodes.
3. Explain the root cause and intended correction briefly, then make the smallest relevant repair. Preserve unrelated changes and artifacts. For transient infrastructure failures, a retry may need no code change or commit. For cancellation by the user, stop. For permissions, credentials, unavailable required files, or a necessary change outside scope, report the blocker rather than spending retries blindly.
4. Validate the correction with a targeted reproduction and appropriate tests; use Bash syntax checking for shell changes. Verify the actual runtime/container where needed. Failed or unavailable validation must be reported accurately; do not label an import check as a successful training run.
5. Review the diff, stage only the repair and its tests, and create a focused local commit. Do not use blanket staging in a dirty repository. Do not include credentials, checkpoints, container images, generated logs, or unrelated user edits. If committing the required repair is blocked, resolve or report that blocker before resubmitting.
6. If the budget permits, record the attempt and commit, then use `sbatch --parsable` with the intended script, directory, and arguments. Capture the returned job ID (and cluster suffix if present), update the ledger, and resume monitoring. A submission rejection consumes an attempt but provides no job to monitor; diagnose it before retrying.

## Success and stopping

Success requires the relevant job/steps to finish successfully, normally `COMPLETED` with `ExitCode=0:0`, plus workload-specific evidence in logs or outputs. For training debugging, verify actual backward propagation and an optimizer update, and checkpoint creation if requested. Model loading, a running allocation, or an empty/no-op epoch is insufficient. If Slurm reports success but the expected work was skipped, treat that as an application failure subject to the same budget.

Stop on verified success, exhausted submissions after the final job finishes, explicit user stop/cancellation, or a blocker requiring user input. Do not automatically cancel a running job merely because monitoring is interrupted.

Conclude with the outcome, all tracked job IDs and states, attempts used out of 3, root causes and fixes, commit SHAs, validation/success evidence, and ledger/log paths. If unsuccessful, identify the remaining error without claiming the workload is fixed.
