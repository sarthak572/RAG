# Retrieval Evaluation Metrics

Reference for the four metrics computed in `rag/evaluation.py` and reported
by `run_eval.py` / `compare_chunking.py`. All four measure **retrieval**
quality only -- which source files came back from `VectorStore.search()`
-- not whether the LLM's final answer was well-written. That split matters:
a wrong final answer can mean either "the right chunk was never retrieved"
or "it was retrieved but the LLM still got it wrong," and these are
different bugs in different stages of the pipeline.

Every metric takes the same two inputs:
- `retrieved`: the source filenames that came back, ranked best-to-worst
  (see `ranked_sources()` in `rag/evaluation.py`)
- `relevant`: the source filename(s) that SHOULD have been retrieved for
  a given question, from `eval/qa_dataset.py`'s hand-written ground truth

## Worked example

Question: *"What is a Process Control Block and what information does it
store?"* -- expects `processes_and_threads.txt`.

Actual result: `retrieved = ['processes_and_threads.txt', 'memory_management.txt']`

## Recall@K

**"Of everything that SHOULD be found, what fraction actually showed up in
my top K results?"**

```
recall@K = |retrieved[:K] ∩ relevant| / |relevant|
```

In the example: 1 expected file, and it's present in the top-5 → **1.0**.
If it hadn't appeared at all in the top 5, Recall@5 would be 0.0.

Recall answers: *is the retriever even capable of finding the right thing,
given a budget of K results?* It says nothing about ranking -- a hit at
position 1 and a hit at position 5 score identically.

## Precision@K

**"Of the K things I retrieved, what fraction were actually relevant?"**

```
precision@K = |retrieved[:K] ∩ relevant| / K
```

In the example: K=5 slots, but only 2 sources came back and 1 of the 2 is
correct → 1/2 = **0.5** (precision here is computed over the number of
distinct sources actually returned, not a full K=5, since fewer than K
distinct files were retrieved).

Precision answers: *how much irrelevant noise is being handed to the LLM?*
A system can have perfect Recall (always finds the right file) while
having poor Precision (also drags in several wrong ones) -- these measure
different things and can move independently.

## MRR (Mean Reciprocal Rank)

**"How far down the ranked list did I have to look before the first
correct result?"**

```
reciprocal_rank = 1 / (rank of first relevant hit)      -- 0 if never found
MRR = average of reciprocal_rank across all questions
```

In the example: the correct file is at rank 1 → RR = 1/1 = **1.0**. If it
had been 2nd, RR = 1/2 = 0.5; 3rd, RR = 1/3 = 0.33.

MRR answers: *does the retriever put the right answer FIRST, not just
somewhere in the top K?* This is the metric Recall can't see -- two
systems can tie on Recall@5 while one always ranks the answer 1st and the
other buries it at 5th every time.

## nDCG@K (normalized Discounted Cumulative Gain)

**"How good is the overall ranking, accounting for both position and
multiple correct answers?"**

```
DCG@K  = sum over ranked results of: 1 / log2(rank + 1), for each relevant hit
IDCG@K = DCG@K of the best possible ordering (all relevant results first)
nDCG@K = DCG@K / IDCG@K
```

In the example: the single relevant file is at rank 1, which is also the
best possible position for it → actual DCG equals ideal DCG → **1.0**.

nDCG generalizes MRR to handle questions with *multiple* correct sources
properly (MRR only looks at the *first* hit and ignores the rest), while
still rewarding earlier positions over later ones via the log2 discount.
It's normalized to always fall between 0 and 1, regardless of how many
relevant sources a question has.

## Why track all four instead of one

They catch different failure modes, and a system can score very
differently across them:

| Metric | Blind to... | Catches... |
|---|---|---|
| Recall@K | ranking order entirely | "did we find it at all" failures |
| Precision@K | whether the *right* one is ranked well | noise / irrelevant retrievals |
| MRR | multiple relevant sources (only sees the first) | "buried the right answer" failures |
| nDCG@K | nothing above -- most complete, but least intuitive | ranking quality with multiple correct answers |

## Ground truth is file-level, not chunk-level -- on purpose

`eval/qa_dataset.py` tags each question with the *file(s)* that should be
retrieved, not the exact chunk. This is deliberate: file-level ground
truth stays valid no matter how a file gets cut into chunks, which is
exactly what lets `compare_chunking.py` reuse the same 24 questions to
compare the fixed/paragraph/recursive chunking strategies against each
other without needing to rewrite the ground truth for every strategy.
