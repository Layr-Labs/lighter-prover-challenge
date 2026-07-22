# Trusted fixtures

`bench.json` is the approved bootstrap fixture from
`Layr-Labs/lighter-auto@d43c6e7158fc3753c23d1c11e43eb8e15006b983`.
That snapshot's `circuit/` tree is
`8603e98de3a6c16addf47d15c78130205d0520c9`, exactly equivalent to the public
`elliottech/lighter-prover@5bbb307dfb26276c48054f2c3ea9dcfe80d3678a`
revision pinned by `benchmark-tools/harness/Cargo.toml`.

- Transactions: 500
- Git blob: `ec94e8c64ccdc14c4a2700f21e7edfb30066131d`
- SHA-256: `d014c969a88bcb0f1673acc410c9e75d1cac53d575463514855050226759c23f`

Keep ranked fixtures outside the editable paths. A rotating private pool is
preferred for future benchmark versions.
