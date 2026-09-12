# Published initialization-selection pilot

Read [研究报告.md](研究报告.md), [facts.json](facts.json), and
[evaluation/directions.csv](evaluation/directions.csv). This compact Git view
includes source, validation/test plans, all aggregate candidate scores, final
per-point predictions, figures and historical audits.

It omits the large `runs/`, `banks/`, and full verification model arrays, as well
as the original full selection seals that refer to those arrays. Retrieve
`rfm-initialization-selection-full.zip` from the
[release](https://github.com/tianxiao1430-jpg/rfm-initialization-study/releases/tag/v1.0.0) for the complete data. Historical checks describe
that complete local study; they must not be rerun against this compact view.

The original `README.md` and Windows `reproduce.ps1` document the historical
workspace. For portable fresh runs use the root `tools/reproduce_selection.py`:

```bash
python tools/reproduce_selection.py --output /path/to/new-directory
```

Run that command from the repository root with NumPy and compatible CUDA PyTorch.
It copies the frozen study sources and old reference engine into a new workspace,
performs preflight, freezes a new execution, runs all methods, evaluates held-out
points, repeats eight selectors and regenerates statistics. SciPy and Matplotlib
are needed for the analysis environment. It never overwrites an existing output.

The original hash manifests refer to original source bytes. Public wrappers are
separate; no measured training implementation was changed for this release.
