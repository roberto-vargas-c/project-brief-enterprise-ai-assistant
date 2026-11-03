# Generated test cases

`python -m ironstore_assistant.testset generate` (Giskard RAGET, see `ironstore_assistant/testset.py`) writes
new questions with reference answers to `generated_cases.jsonl` in this folder, one JSON object per line:

```json
{"question": "...", "category": "generated: simple", "expected_sources": ["IronStore_Remote_Work_Policy.pdf"],
 "reference": "...", "answerable": true, "history": [], "origin": "giskard:simple",
 "reviewed": false, "reference_context": "the chunk the question was written from"}
```

Generated questions can be wrong, vague or about the wrong document. Before they count:

1. Read each question with its `reference_context`.
2. Fix the question or the `reference`, add `must_include` keywords if useful, or delete the line.
3. Set `"reviewed": true`.

`python -m ironstore_assistant.evaluate` uses only reviewed lines (add `--include-unreviewed` to try the rest).
Running `generate` again appends new questions and keeps the ones already reviewed.
