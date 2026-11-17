# Kimi build6 schema preflight integration

The SGLang Kimi schema helper is applied to the validated effective tool collection before template rendering and before structural-tag capability probing. Both global and message-position declarations retain their positions while consuming the same canonical parameter schema as the parser/guidance projection.

KimiSchemaCanonicalError is converted to PreprocessError at this early boundary, so the grammar builder's generic ValueError fallback cannot silently turn rejected recursive schemas into unconstrained generation. The compatibility policy is Kimi-only. The paired helper implementation and tests are in the SGLang repository.

Covered build6 failures: eight unique Walle conflict/recursion cases, each with streaming and nonstreaming execution (IDs 099–114). Canonical relaxation is an explicit Walle contract, not a general JSON Schema equivalence. See workspace `engine侧修复` records for original assertions, precise case IDs, paired commits and observed results.
