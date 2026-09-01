# Decision Trace milestone inventory

The renderer has one fixed graph per processing pipeline. A route selects edges in
that graph; state updates only advance the cursor by one milestone and never
replace geometry already shown.

| Pipeline | Route | Ordered milestones |
| --- | --- | --- |
| Ingest | success | raw -> triage -> target -> generate -> authority -> result -> change -> apply -> publish -> readback -> complete |
| Ingest | noop | raw -> triage -> target -> generate -> authority -> result -> change -> readback -> complete |
| Ingest | hold | raw -> triage -> target -> generate -> authority -> result -> hold |
| Ingest | retry | raw -> triage -> target -> generate -> authority -> result -> generate -> authority -> result -> change -> apply -> publish -> readback -> complete |
| Recall | success | search -> rerank -> authority -> result -> commit -> readback -> complete |
| Recall | hold | search -> rerank -> authority -> result -> hold |
| Audit | success | select -> inspect -> consensus -> result -> report -> complete |
| Audit | hold | select -> inspect -> consensus -> result -> hold |
| Improve | success | discover -> generate -> verify -> result -> apply -> readback -> complete |
| Improve | hold | discover -> generate -> verify -> result -> hold |
| Repair | success | detect -> local_fix -> verify -> result -> readback -> complete |
| Repair | hold | detect -> local_fix -> verify -> result -> hold |
| Repair | escalate | detect -> local_fix -> verify -> result -> escalate -> verify -> result -> readback -> complete |
| Typed Graph | success | discover -> extract -> verify -> consolidate -> evaluate -> result -> promote -> readback -> complete |
| Typed Graph | hold | discover -> extract -> verify -> consolidate -> evaluate -> result -> hold |

Acceptance coverage:

- 6 pipeline graphs
- 15 routes
- 112 milestone frames
- 112 screenshots at 2048 x 1200
- one geometry hash per pipeline across every state
- visible cursor equals requested cursor in every frame
- every SVG edge endpoint meets its source and target node
- no unrelated SVG edges cross
- no path/label or label/label intersections
- browser stepper advances exactly one milestone every 500 ms
