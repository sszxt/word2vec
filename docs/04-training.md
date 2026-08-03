# Training procedure

Implementation: `src/word2vec/train.py`. Builds on `docs/02-architectures.md`
(what's being predicted) and `docs/03-hierarchical-softmax.md` (the loss).

## What the paper specifies

Section 4.2: plain SGD with backpropagation, starting learning rate 0.025,
decreased linearly so it reaches (approximately) zero at the end of the last
training epoch. Section 2.3 notes their large-scale runs use "mini-batch
asynchronous gradient descent" across ~100 model replicas via a framework
called DistBelief — so even the paper's own training is not literally one
example at a time once it's running at scale.

We implement single-machine minibatch SGD: shuffle all (context, target)
pairs for the current epoch, and step once per minibatch rather than once per
example. This is directly analogous to what Section 2.3 describes, just
without the distributed replica/parameter-server machinery.

## A batching pitfall worth understanding, not just avoiding

The first working version of this training loop used the natural PyTorch
pattern: `loss = model(batch)` returns the *mean* per-example loss over the
minibatch, and `loss.backward()` backpropagates that mean directly. Training
ran, the loss started at the expected random-initialization value, dropped
slightly in the first few batches, and then sat completely flat for the rest
of the epoch (see git history for the exact numbers).

The cause: the paper's SGD takes one gradient step per training example. In a
mean-reduced minibatch of size `B`, the gradient of the mean loss has the same
*expected direction* as a single example's gradient, but its effect on the
weights after one `optimizer.step()` is only equivalent to *one* per-example
update, not `B` of them. Batching this way for GPU throughput, without
changing anything else, silently reduces the total number of effective
gradient updates applied over an epoch by a factor of `B` — with `B=2048`,
that's 2048 previously per-example updates in the paper's version of training
collapsed into a single effective update in ours. Over one epoch, the model
barely moves.

Fix, in `train.py`: backpropagate `loss * batch_size` instead of `loss`
directly (i.e. use the *sum* of per-example losses, scaled implicitly, rather
than the mean), while still logging the *mean* for a human-readable,
batch-size-independent loss curve. This restores gradient magnitude to what
`B` sequential per-example SGD updates at the same learning rate would have
produced, so `--lr` keeps meaning "the paper's per-example learning rate"
regardless of what batch size is chosen for GPU efficiency. Verified
empirically on a 2M-token / 3-epoch CBOW run: loss dropped from 7.50 to 6.67
and kept decreasing smoothly, with no instability, after the fix — versus
completely flat before it.

This is the kind of bug that a bare "the loss curve looks reasonable-ish"
glance would miss (it decreases briefly, then is merely quiet, not obviously
broken) but that shows up immediately once you check it against the
paper-implied training dynamics rather than just "does the number go down at
all."

## A second pitfall: SGD divergence without momentum

Fixing the scaling above exposed a second issue, this time specific to
Skip-gram. With `batch_size=2048`, CBOW trained fine (loss dropping smoothly,
4.2% total analogy accuracy), but the Skip-gram checkpoint's word vectors
were entirely NaN.

Direct measurement (`clip_grad_norm_` with an effectively infinite max norm,
just to read the value without clipping) showed the mechanism: Skip-gram's
global gradient norm was a normal ~75-150 for roughly its first 500 steps --
statistically indistinguishable from CBOW's -- then grew to ~2 billion by
step ~2000 and stayed there, with the loss simultaneously exploding into the
trillions. This is plain SGD's classic failure mode: with no momentum or
per-parameter adaptive scaling (Adam, Adagrad) to damp it, one unusually
large update can overshoot far enough that the *next* batch's gradient
through the now-displaced vector is even larger, compounding exponentially
within a couple thousand steps. CBOW's epoch is only 8,164 steps (16.7M
pairs / batch_size) versus Skip-gram's 89,794 (11x more pairs from the same
corpus), so in this run CBOW's shorter epoch simply ended before it happened
to hit a destabilizing sequence of batches -- not because CBOW is immune to
it.

Fix: clip the global gradient norm to just above the observed healthy range.
A clip of 150 caps the worst-case single step without touching normal ones,
breaking the compounding loop before it can start. Verified empirically on
the exact failing configuration (Skip-gram, dim=100, full text8,
batch_size=2048, 1 epoch): no NaN, 17.9% total analogy accuracy. Verified
the same setting doesn't cost CBOW anything: 4.2% total accuracy clipped,
identical to unclipped.

That 150 was right — but only for the batch size it was measured at, which
took a while to notice. See "A third pitfall" below.

A first attempt at this fix used `max_norm=5.0`, reasoning (incorrectly)
that the paper's implicit per-example SGD should only ever need very small
per-step gradients. That value silently crushed CBOW's real training signal
along with the instability -- CBOW's own healthy gradient norms sit at
~75-150, so a clip of 5 throttled essentially every step, and total accuracy
collapsed from 4.2% to 0.3%. The lesson: measure the actual healthy gradient
scale before picking a clipping threshold, rather than picking a
small-sounding number and assuming smaller is safer.

## A third pitfall: a clipping threshold that only works at one batch size

The two fixes above interact, and the interaction hid a bug for a while.

Because the loss is scaled by `len(batch)` before `backward()` (first pitfall),
the gradient norm is *proportional to batch size*. But the clip introduced by
the second pitfall was a fixed constant, 150, calibrated by measuring norms at
`batch_size=2048`. A fixed threshold against a batch-size-dependent quantity is
only meaningful at the batch size where it was measured.

Measured pre-clip global gradient norms, CBOW, 300 steps each:

| batch size | median norm | % of batches clipped at 150 |
|---|---|---|
| 512 | 24 | 0% |
| 2048 | 75 | 0% |
| 8192 | 407 | **98%** |

At `batch_size=8192` the "safety" clip renormalizes almost every gradient in
training, silently throttling learning to a crawl. Nothing crashes, nothing
NaNs, the loss still decreases — it just decreases from behind an invisible
handbrake. Anyone who changed `--batch-size` for GPU-memory reasons got a
quietly different algorithm.

Fix, in `train.py`: derive the threshold per-example rather than fixing it.
`_CLIP_PER_EXAMPLE = 150/2048`, so `auto_grad_clip()` reproduces the original
calibration *exactly* at batch 2048 and scales correctly everywhere else.
Passing `--grad-clip` explicitly still overrides it. Backward compatibility is
verified against the previously published result: CBOW dim=100 full data at
seed 0 still gives exactly 3.67 / 4.55 / 4.18.

The general lesson is narrower than "clip gradients" and more useful: a
threshold is a measurement, and a measurement is only valid under the
conditions it was taken. The second pitfall's own lesson — measure the healthy
gradient scale before choosing a clip — was correct but incomplete, because it
did not ask *what that scale depends on*.

## CBOW's context gradient: where the paper stops specifying

CBOW averages its context word vectors before scoring (`docs/02`). The
mathematically correct derivative of that average gives each of the `C` context
words `1/C` of the upstream gradient, which is what `.mean(dim=1)` produces
under autograd and what this implementation originally did.

The reference `word2vec.c` does something else. It averages going forward
(`neu1[c] /= cw`) and then adds the accumulated gradient `neu1e` to every
context word **undivided** — so each context word receives `C` times more
update than the true gradient of its own forward pass would give. That is not
a derivative of anything; it is effectively a `C`-fold larger learning rate on
the input embeddings.

The paper's text never specifies the backward pass, so there is no way to
settle this from the paper alone. Since the published numbers were produced by
the authors' code, that code is the tiebreaker, and reproduction runs use the
`sum` rule (`--context-grad`, default `mean`; `REPRO_CONTEXT_GRAD` in
`scripts/run_experiments.py`).

It matters more than it sounds. Over three seeds, CBOW dim=100 on full text8:

| rule | total accuracy |
|---|---|
| `mean` (true gradient) | 4.35% ± 0.18 |
| `sum` (reference behaviour) | **7.49% ± 0.27** |

This is also why the third pitfall took so long to find. The `sum` rule raises
gradient norms roughly 4.5x, so under the old fixed clip of 150 it was almost
entirely clipped away and measured *worse* than `mean` — 3.8% against 4.2%.
The first conclusion drawn from that was "the reference rule doesn't help,"
which was exactly backwards: the experiment had measured the clip, not the
rule. Neither fix is visible without the other.

## Learning rate schedule

`current_lr = max(base_lr * (1 - step / total_steps), base_lr * 1e-4)`, where
`step` is the global minibatch counter across all epochs and `total_steps` is
`steps_per_epoch * num_epochs`. This linearly decays from `base_lr` to
(approximately) zero across the full training run, matching Section 4.2. We
floor it at `1e-4 * base_lr` rather than letting it hit exactly zero, matching
the original word2vec.c implementation, so the very last few minibatches
still receive a (tiny) update rather than a no-op.

## Weight initialization

- Input/output word embeddings: uniform in `[-0.5/D, 0.5/D]`.
- Hierarchical-softmax internal-node vectors: all zeros.

Both match the original word2vec.c reference implementation. The
zero-initialized node vectors are worth noting because of a subtle
consequence: on the very first minibatch, every score `input_vector .
node_vector` is exactly 0 regardless of the (randomly initialized) input
vector, so the *input* embeddings receive zero gradient on step 1 — only the
node vectors move. Input embeddings start receiving signal from step 2
onward, once the node vectors they're compared against are no longer zero.
This is expected, not a bug.

## Data regeneration per epoch

For Skip-gram, the dynamic window size `R` (docs/02-architectures.md) is
resampled per position on every call to `generate_skipgram_pairs`. We
regenerate the full pair set at the start of every epoch rather than reusing
the first epoch's pairs, so multi-epoch Skip-gram training sees a fresh
random window each pass, matching the paper's description of the sampling
procedure rather than freezing one arbitrary sample of it for the whole run.
