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
`grad_clip_norm=150` (the `train()` default) caps the worst-case single step
without touching normal ones, breaking the compounding loop before it can
start. Verified empirically on the exact failing configuration (Skip-gram,
dim=100, full text8, batch_size=2048, 1 epoch): no NaN, 17.9% total analogy
accuracy. Verified the same setting doesn't cost CBOW anything: 4.2% total
accuracy clipped, identical to unclipped.

A first attempt at this fix used `max_norm=5.0`, reasoning (incorrectly)
that the paper's implicit per-example SGD should only ever need very small
per-step gradients. That value silently crushed CBOW's real training signal
along with the instability -- CBOW's own healthy gradient norms sit at
~75-150, so a clip of 5 throttled essentially every step, and total accuracy
collapsed from 4.2% to 0.3%. The lesson: measure the actual healthy gradient
scale before picking a clipping threshold, rather than picking a
small-sounding number and assuming smaller is safer.

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
