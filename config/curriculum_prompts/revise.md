Revise G1 tasks using saved observations and the requested quotas. Observations are fallible.
G1 contains stationary rigid objects/scenes and camera orbit, dolly, or lateral translation.
Change exactly one axis per edited prompt: object/scene, camera path, viewpoint range, or clutter.
Record the changed axis in attributes.edit_axis and the intended change in attributes.edit.
Preserve static-scene assumptions. Do not certify feasibility or physical correctness.
For edits, inherit the parent's lineage and partition, use its prompt_id as parent_id,
and assign a new prompt_id and the requested next round. Initial/new prompts establish a new lineage.
Use only the supplied training parents and aggregate development feedback; never infer test tasks.
Return only a JSON array of PromptRecord objects: prompt_id, lineage_id, parent_id, split,
text, subtype, attributes, source, round_index. Match the exact requested source counts.
No markdown or commentary. Do not silently lower requirements or change task domains.
