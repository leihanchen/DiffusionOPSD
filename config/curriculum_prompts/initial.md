Create initial G1 video tasks from the supplied domain contract, without task examples.
G1 contains stationary rigid objects/scenes and camera orbit, dolly, or lateral translation.
Describe a feasible scene, explicit camera path, and invariant object identity/geometry.
Exclude moving objects, deformation, gravity/contact claims, and impossible simultaneous camera demands.
Respect the requested split, round, count, sources, and enabled subtypes. Generate distinct scene lineages.
Return only a JSON array of PromptRecord objects with exactly these fields:
prompt_id, lineage_id, parent_id, split, text, subtype, attributes, source, round_index.
For initial/new records, parent_id is null and lineage_id equals the new prompt_id.
Attributes map strings to strings. No markdown, commentary, or semantic accuracy certification.
