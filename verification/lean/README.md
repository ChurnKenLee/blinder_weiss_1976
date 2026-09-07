# Lean proof environment

This package pins mathlib release `v4.33.1` to commit
`0df444a360eaa60ab8c11dca51a86af692955474`. Its `lean-toolchain` file selects
`leanprover/lean4:v4.33.1`, matching that mathlib release. The checked-in
`lake-manifest.json` locks transitive dependency commits.

Install [elan using Lean's official instructions](https://lean-lang.org/install/manual/).
Then, from this directory:

```sh
lake exe cache get
lake build
```

Lean selects the compiler from `lean-toolchain`; do not replace the pinned
mathlib revision with a moving branch when reproducing these proofs. Mathlib's
[dependency guide](https://github.com/leanprover-community/mathlib4/wiki/Using-mathlib4-as-a-dependency)
explains dependency installation and the binary cache. Downloaded dependencies
and build products stay in the ignored `.lake/` directory.

The current temporary notebook container uses an isolated elan installation:

```sh
ELAN_HOME=/tmp/bw-elan /tmp/bw-elan/bin/lake build
```

Proof modules live under `BlinderWeiss/`, in namespace `BlinderWeiss`.
`Foundations.lean` contains only a compiler smoke theorem; its successful build
does not verify any claim from the paper. Other modules state their hypotheses
explicitly. Their paper correspondence and the remaining claims are tracked
outside this build package in `../CLAIMS.md`. The source transcription and
provenance are stored in `../source/`.
