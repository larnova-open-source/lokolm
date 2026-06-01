# Contributing to lokoLM

Thanks for your interest in lokoLM! lokoLM is **open source**, and contributions are
welcome — but the project has a deliberately **narrow contribution scope**. Please read this
guide before opening a pull request or issue.

**Project author / maintainer:** Mahmud Suberu — Founder & CEO of Larnova
([LinkedIn](https://www.linkedin.com/in/mahmud-adinoyi-684020235/)).

> This is the copy living in the lokoLM **model** repository
> ([lokolm](https://github.com/larnova-open-source/lokolm)). The same guide is also published
> in the [lokolm-docs](https://github.com/larnova-open-source/lokolm-docs) repository.

---

## What you can contribute

lokoLM v1 is a **minimal, educational decoder-only Transformer**. The model code is kept
small and readable on purpose, and its architecture is curated by the maintainer to stay
that way. So contributions are focused on **training**, in two forms:

1. **Training data** — raw text corpora to pretrain the model on.
2. **Training checkpoints** — weights (`ckpt.pt`) you produced by training lokoLM.

That's the contribution surface. Architecture, tokenizer, and feature changes (RMSNorm,
SwiGLU, RoPE, GQA, MoE, BPE, fine-tuning, agents) are tracked on the
[roadmap](https://github.com/larnova-open-source/lokolm-docs/blob/main/content/roadmap.md)
(in the lokolm-docs repository) and owned by the maintainer — please **open an issue to
discuss** before sending code for those, rather than a direct PR.

---

## ⚠️ Data must be UNLABELLED

This is the most important rule.

lokoLM is **pretrained with self-supervised next-token prediction**: it learns language by
predicting the next byte in raw text. It has **no labels, no annotations, no input→output
pairs**. The training loop ([train.py](train.py)) just reads a stream of bytes and learns to
continue it.

**So all contributed data must be plain, unlabelled text.**

✅ **Yes — unlabelled raw text:**
- Books, articles, documentation, transcripts, code, web text, etc.
- A single UTF-8 text file (or many concatenated into one) — just the *content*.

❌ **No — labelled / structured data:**
- Classification datasets (text + a category label)
- Question→answer or instruction→response pairs
- Chat/conversation datasets with role labels
- Anything in a `{"input": ..., "label": ...}` / CSV-with-label-column format

> Labelled and instruction-style data **are** valuable — but they belong to **fine-tuning**,
> which is a **v2 roadmap** item ([Track B](https://github.com/larnova-open-source/lokolm-docs/blob/main/content/roadmap.md)),
> not to v1 pretraining. If you submit labelled data for v1, it will be declined with a
> pointer back here. When fine-tuning lands, we'll open a separate contribution path for it.

If you're unsure whether your data counts as "labelled," ask yourself: *does each example
carry an answer/category/target that's separate from the text itself?* If yes, it's labelled
— don't submit it for v1.

---

## Contributing training data

**Format.** Plain UTF-8 text. The trainer reads a file named `input.txt` next to
`train.py`:

```powershell
# drop your corpus in as input.txt, then:
python train.py
```

**Guidelines:**
- **One file, raw content only.** No headers, labels, or metadata columns mixed in.
- **License it.** Only contribute text you have the right to share. State the **source and
  license** of the corpus in your PR/issue. Public-domain, permissively licensed, or
  your-own text is ideal. Do **not** submit copyrighted or scraped data you can't
  redistribute.
- **No private or personal data (PII).** No names+contact info, credentials, health/financial
  records, or anything that shouldn't be public. The model can memorize and reproduce its
  training data.
- **Quality over quantity.** Clean, well-formed text trains better than large, noisy dumps.

**How to submit.** Corpora are usually large and are **not committed to git** (see
[.gitignore](.gitignore)). **Don't** paste a big file into a PR. Instead, open an **issue**
with:
- a short description of the corpus,
- its size, language(s), source, and license,
- a **download link** (e.g. a release asset, Hugging Face dataset, or cloud bucket).

---

## Contributing a training checkpoint

A checkpoint is the weights you got from training lokoLM. The trainer saves it as `ckpt.pt`,
a dict holding both the weights **and** the model config:

```python
checkpoint = {"model": state_dict, "config": {...}}   # written by train.py
```

**Guidelines:**
- **Must load with the current code.** It has to be loadable by [sample.py](sample.py) — i.e.
  the `{"model", "config"}` format produced by the current [train.py](train.py). Strip the
  `_orig_mod.` prefix yourself if you trained with `torch.compile`, or note it.
- **Document the run.** Include the **config** (or confirm it's the saved default), what
  **data** it was trained on (and that data's license — same rules as above: unlabelled,
  shareable), iterations/steps, and the final train/val loss.
- **Provenance matters.** A checkpoint inherits the licensing and content of its training
  data. Don't contribute weights trained on data you couldn't have contributed directly.

**How to submit.** Checkpoints are large binaries and are **gitignored** — don't commit them.
Open an **issue** with a **download link** and the details above.

---

## Submitting

- **Data & checkpoints → open an issue** with a link and the required details (above).
  Maintainers will review and integrate.
- **Small fixes** (typos, docs, obvious bugs) → a PR is fine.
- **Architecture / feature ideas** → open an issue to discuss; these are maintainer-owned and
  follow the [roadmap](https://github.com/larnova-open-source/lokolm-docs/blob/main/content/roadmap.md).

---

## License

lokoLM is released under the [MIT License](LICENSE). By contributing, you confirm you have
the right to share what you submit and agree it may be used under that license.

Note that **trained checkpoints and training data carry the license of the underlying data**
— MIT covers the lokoLM code, not the corpora you train on. Always state your data's source
and license (see above).

---

Questions? Open an issue. Thanks for helping lokoLM learn. — *[Mahmud Suberu, Larnova](https://www.linkedin.com/in/mahmud-adinoyi-684020235/)*
