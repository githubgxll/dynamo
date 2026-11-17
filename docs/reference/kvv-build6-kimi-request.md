# Kimi build6 request processing

The Dingo SGLang frontend now keeps separate projections for original-position template declarations and effective parser/grammar tools. Dynamic system/developer declarations are validated before generation, including shape, content conflicts, required fields, the 256-character Kimi name contract and duplicates across declarations. Required/named choices and both inline/pool postprocessors use the effective set, without moving dynamic declarations into the global template position.

The Kimi-only request helper preserves raw global tool fields instead of injecting parser-model defaults. It forwards explicit tool-choice, response-format and resolved effort controls to the Kimi encoder. `none` suppresses calls without removing tool declarations; explicit `none` without tools is distinguished from an omitted choice. Reserved template keys cannot silently replace validated declarations. Other model paths retain their existing processing.

Dynamic-tools build6 cases: 001–003,005–011,013–020,022,024,026–029,031–032,034,036–040,042,044–045,047,049–052,055,059,063,067–068,071,074,076,078–079,082,084,087 (54 executions, identified unambiguously in the workspace manifest). Prompt tool declaration and control-message cases also depend on this adapter.

The inline and process-pool paths propagate effective tools to both parser creation and output validation. Shared unit regressions cover these request contracts; real KVV results are kept under workspace `0ai/kimi-k31/kvv测试/engine侧修复`. Gateway-dropped fields cannot be recovered by this adapter. Kimi nested effort additionally needs the paired native normalization repair because the old native layer dropped the original thinking object.
