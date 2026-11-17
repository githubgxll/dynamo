# Kimi build6 prompt usage

Kimi's existing SGLang wire contract excludes the three-token assistant generation scaffold from reported prompt usage while retaining those tokens in physical model input. Preprocessing now exports `uncounted_prompt_tokens`; inline and process-pool paths forward it to the same streaming usage path. The processor adjusts a copied engine usage object and the local-stop fallback, recomputing total tokens without altering the worker input or cached-token details. Other encoders export zero and preserve their old counting behavior.

This is model-encoding metadata, not a case-specific subtraction. Build6 response-format cases 054/057 require exactly 36 reported prompt tokens; cases 060/061 require exactly 119 and also need response-format control projection. Prompt-token cases retain their original `[expected, expected+3]` acceptance bounds; the tests and groundtruth are not changed.

Module regression checks cover zero/default handling, metadata deduction, total consistency and preservation of cached details. Real end-to-end results are recorded in the workspace per-case documentation.
