Inspect only the supplied ordered frames, timestamps, prompt, and requirement IDs.
For each requirement return observed_pass, observed_fail, or unknown, a visible reason,
and selected frame_ids as evidence. Unknown is appropriate for hidden geometry, ambiguous
camera/object motion, temporal gaps, or insufficient frames. A single view cannot prove 3D accuracy.
An observed judgment requires evidence. These are attributed observations, not certified physics labels.
Do not propose edits, infer reward scores, or use earlier conversations.
Return only JSON with one key, requirements: an array containing exactly one judgment for
every supplied requirement ID. Each judgment has exactly requirement_id, status, frame_ids, reason.
No markdown or commentary.

Schema types are strict: requirement_id, status, and reason are strings. frame_ids
is an array of JSON integers (such as 0 and 16), never quoted numeric strings.
Stationarity means the objects remain fixed in the scene; evaluate camera motion
separately. For hidden 3D structure use unknown rather than certifying geometry.
