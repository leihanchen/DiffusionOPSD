`train.txt` and `test.txt` list one text prompt per line, each describing motion explicitly (camera pan, orbit, dolly, truck, pedestal, or rigid object translation/rotation in a static scene). The two splits are disjoint: training prompts cover varied indoor/outdoor subjects and motion verbs, while `test.txt` is curated in the spirit of WorldScore and VBench-2.0 camera-control and 3D-consistency evaluation prompts (fixed trajectories, parallax and depth cues, no duplicate wording from train).

The Wan trainer can evaluate the policy already in memory. `config.eval_freq` is
the interval in completed optimizer updates; `0` disables evaluation. When enabled,
evaluation also runs before the first update and after the final update. The base
Wan config disables it; the four-update Wan2.2 debug config evaluates at updates
0, 2, and 4.

`eval_num_prompts` selects the first nonempty prompts from `eval_prompts` (default:
8). Each uses seed `eval_seed + prompt_index` (default seed: 0) at every evaluation.
The starting policy supplies the baseline, including any initial LoRA weights.
Scores include geometry, identity, motion, VideoReward VQ/MQ/TA/Overall, changes
from baseline, and the existing relative VQ/MQ score `p_q`. This score is a sigmoid
of the worse VQ/MQ gap; it is not an empirical win rate. Repeated monitoring on this
subset should be treated as validation, not an independent final test.

Scalar metrics appear under `eval/` in offline W&B, stdout, and `metrics.jsonl`.
Per-prompt scores and seeds are saved to `<logdir>/eval/update_XXXXXX.json`.
`eval_num_videos=2` saves the same two prompt examples as MP4s beside those files
and attaches them to W&B. `eval_video_fps=8` controls playback; set
`eval_num_videos=0` to log scalars only. Video encoding uses the imageio/FFmpeg
packages included in the training container. Evaluation does not load another
checkpoint or keep baseline videos on the GPU.
