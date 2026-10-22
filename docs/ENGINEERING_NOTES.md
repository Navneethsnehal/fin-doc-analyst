# Engineering notes

These notes describe the code in this repository. There is no ADR directory. Decisions below are the ones encoded in the modules. Numbers come from reading the code or from `pytest` and `python scripts/run_eval.py` run during this review (Python 3.12.3, fake LLM, hash embeddings, memory index).

## Files

| Path | Role |
| --- | --- |
| `src/fin_doc_analyst/__init__.py` | Package version `0.1.0`. |
| `src/fin_doc_analyst/config.py` | Settings from the environment and `.env`. Resolves `auto` to a concrete LLM and embedding provider. Turns LangSmith tracing on or off by setting process env vars. |
| `src/fin_doc_analyst/schemas.py` | Pydantic models shared by ingest, retrieval, the graph, and the API: `Chunk`, `DocumentInfo`, `Citation`, `RiskFlag`, `ToolTrace`, `AnalysisResult`. |
| `src/fin_doc_analyst/textutil.py` | Tokenizer, light suffix stemmer, synonym expansion, sentence split, currency regex, and the overlap score used by grading and the fake provider. |
| `src/fin_doc_analyst/logging_config.py` | structlog JSON logs. A processor runs `redact` on top-level string fields. |
| `src/fin_doc_analyst/ingestion/pdf.py` | pypdf text extraction, one string per page, 1-based page numbers. Any pypdf failure becomes `PdfParseError`. |
| `src/fin_doc_analyst/ingestion/chunker.py` | Splits each page at `Section:` headings, then into pieces of at most 700 characters with 80 characters of overlap. Chunks do not cross pages. |
| `src/fin_doc_analyst/ingestion/pipeline.py` | Builds a `doc_id` from the filename and returns `DocumentInfo` plus chunks. `synthetic` is set true here for every file this function ingests. |
| `src/fin_doc_analyst/retrieval/embeddings.py` | `HashEmbedder` (default), `OpenAIEmbedder`, `OllamaEmbedder`. Query and document vectors are the same function for the hash and Ollama embedders. |
| `src/fin_doc_analyst/retrieval/hybrid.py` | BM25 plus dense scores, reciprocal rank fusion (`k = 60`), then a lexical rerank. The rerank keeps items at least 0.55 times the best score, capped by the requested `k`. |
| `src/fin_doc_analyst/retrieval/memory.py` | Default store. Rebuilds `HybridIndex` on add or delete. A lock guards the chunk list. |
| `src/fin_doc_analyst/retrieval/pgvector_store.py` | Postgres + pgvector for the dense half. BM25 is rebuilt in process from the loaded chunk text. HNSW index creation is skipped if the server rejects it. |
| `src/fin_doc_analyst/retrieval/factory.py` | `memory` or `pgvector`. Anything else raises. |
| `src/fin_doc_analyst/llm/base.py` | `LLMClient` protocol (`complete_json`) and `LLMError`. |
| `src/fin_doc_analyst/llm/prompts.py` | System prompts for route, grade, and answer. They tell the model to return JSON and to treat excerpts as data. |
| `src/fin_doc_analyst/llm/parsing.py` | Pulls the first JSON object out of model text, including fenced blocks. `coerce_bool` maps JSON booleans and the strings `true` / `false`. |
| `src/fin_doc_analyst/llm/fake.py` | Keyword router used by tests and CI. Its grade and answer methods exist but the graph does not call them: the fake provider is special-cased in the nodes. |
| `src/fin_doc_analyst/llm/openai_client.py` | Chat completions at temperature 0 with `response_format` JSON. |
| `src/fin_doc_analyst/llm/anthropic_client.py` | Messages API at temperature 0. Text blocks are concatenated and parsed as JSON. |
| `src/fin_doc_analyst/llm/ollama_client.py` | `POST /api/chat` with `format: json` and temperature 0. |
| `src/fin_doc_analyst/llm/factory.py` | Builds one of the four clients from settings. |
| `src/fin_doc_analyst/guardrails/injection.py` | Fixed regular expressions (ignore-previous-instructions, admin mode, reply with APPROVED, and the others in `INJECTION_PATTERNS`). `neutralize` replaces matches with `[filtered-instruction]`. |
| `src/fin_doc_analyst/guardrails/pii.py` | Replaces dashed SSNs, labeled SSNs, labeled account and routing numbers (digits may contain spaces or hyphens), and unlabeled 10–17 digit runs. |
| `src/fin_doc_analyst/guardrails/grounding.py` | An answer is grounded when every number in the body is in the context or tool JSON, and the remaining token overlap is at least 0.72. Lines that start with `Source:` are ignored. |
| `src/fin_doc_analyst/agent/state.py` | `AgentState` and the refusal strings. |
| `src/fin_doc_analyst/agent/tools.py` | `sum_currency` and `amortized_payment`. Both quantize with `ROUND_HALF_UP` to cents. |
| `src/fin_doc_analyst/agent/answer.py` | Extractive answers for the fake provider, citation formatting, and the rules that choose tool arguments from chunk text. |
| `src/fin_doc_analyst/agent/risk.py` | Regex scan for variable rate, auto-renewal, exclusion, acceleration, prepayment, penalty, and fee. Skips the Deposits, Withdrawals, and Account Summary sections and dated posting lines. |
| `src/fin_doc_analyst/agent/nodes.py` | The graph nodes. Fact answers for a non-fake provider go through `ANSWER_SYSTEM`. Grading for a non-fake provider goes through `GRADE_SYSTEM`. |
| `src/fin_doc_analyst/agent/graph.py` | Wires the nodes. `compute` and `risk` run even when `grade` kept nothing. `ground` routes an ungrounded draft to `refuse`. |
| `src/fin_doc_analyst/agent/service.py` | Loads samples and uploads, invokes the graph, redacts the result. Strips retrieved text from the response unless `include_context` is set. |
| `src/fin_doc_analyst/api/app.py` | FastAPI routes. The module-level `app` builds a service on startup if one was not injected. |
| `src/fin_doc_analyst/evals/metrics.py` | Faithfulness, fact recall, context precision, context recall. No judge model. |
| `src/fin_doc_analyst/evals/thresholds.py` | Floors CI compares against. |
| `src/fin_doc_analyst/evals/runner.py` | Loads the golden file, runs the service, prints the table, writes JSON, exits 1 on a miss. |
| `apps/streamlit_app.py` | Calls the in-process service, or the API when `API_BASE_URL` is set. |
| `scripts/generate_samples.py` | Writes the four PDFs with reportlab. The stated loan payment is `amortized_payment("18400.00", "0.0749", 60)`. |
| `scripts/run_eval.py` | Calls `evals.runner.main`. |
| `data/evals/golden.json` | 15 questions plus 4 injection questions, with expected route, refusal, facts, and evidence strings. |
| `tests/conftest.py` | Forces the fake provider and memory backend before the app is imported. The `service` fixture loads `data/samples/`. |
| `Dockerfile`, `docker-compose.yml` | API image and a Compose stack with pgvector, the API, and Streamlit. |
| `.github/workflows/ci.yml` | `ruff check`, `ruff format --check`, `pytest`, `python scripts/run_eval.py` on Python 3.11. |

## Decisions and tradeoffs

**Page-bounded chunks.** `chunk_pages` never joins two pages. The page on a citation is the page index from pypdf. The cost is that a sentence near the edge of a 700-character window is repeated in the next window (overlap 80). Sums dedupe identical sentences. Retrieval can still return both copies; the rerank may keep only one.

**Hybrid search without a cross-encoder.** BM25 is there for exact tokens such as fee names. The hash embedder is there so a paraphrase can still share buckets with a synonym-expanded query. Reciprocal rank fusion merges the two rankings. The rerank is a weighted lexical overlap plus the fused score (`0.62` lexical, `0.38` normalized fusion, `0.12` if every query token is in the passage). CI does not download a model. Paraphrases outside `SYNONYMS` in `textutil.py` depend on token overlap.

**One graph, and a fake client that does not generate prose.** Hosted providers implement `complete_json`. The fake provider's `complete_json` only classifies. For `provider == "fake"`, grading and answering use `overlap_score` and copy sentences that score at least 0.75. That makes the golden file deterministic. It also means a perfect eval score does not measure a hosted model. Hosted answering is still checked for grounding after the model returns.

**Arithmetic stays in tools.** `plan_tools` picks `sum_currency` or `amortized_payment` from the question and the chunk text for every provider. The model is not given a tool-calling API. The benefit is that cents do not depend on the model. The cost is that phrasing the planner does not recognize never reaches a tool. `classify_question` and `plan_tools` are separate keyword lists; they have to be kept in sync.

**Risk scan ignores top-k.** `analyze_risk` scans every in-scope chunk that is not an injection match. A clause that retrieval would have missed can still be flagged. A clause that does not match `PATTERNS` is not flagged. Posting tables are skipped so a dated fee line is not also a "clause."

**Injection fails closed, with a fixed phrase list.** A matching user turn never retrieves. A matching chunk is dropped in `grade` and is also excluded from `compute` and `analyze_risk`. `neutralize` only runs on text sent to a hosted grader or answerer. The list is not a general detector. A payload that does not match `INJECTION_PATTERNS` is ordinary document text.

**Redact on the way out.** `AnalystService.ask` redacts the answer, citation excerpts, and risk text. The log processor uses the same `redact` function on top-level strings. Nested dicts in a log event are not walked. Unlabeled numbers shorter than 10 digits are left alone unless a label such as `account` or `SSN` is next to them.

**Grounding is a node.** After `answer` or `analyze_risk`, `is_grounded` checks numbers and token overlap. Failure routes to `refuse`, which replaces the draft with `UNGROUNDED` (the same sentence as `NOT_FOUND`). The `grounded` flag on that refusal is then set true, meaning the text returned to the user is the canned refusal. `refusal_reason` is `ungrounded`.

**Judge-free evals.** Faithfulness reuses `grounding_score`. Answer relevance and context precision/recall are substring checks of the hand-written facts and evidence (`contains_fact` lowercases and strips `$` and commas). Thresholds sit under the fake-provider run (`0.80`, `0.85`, `0.70`, `0.80`, and `1.00` for the safety and routing metrics). The fake run is not given room to miss a refusal.

**Memory index by default.** Tests and the eval script never open Postgres. pgvector is there for Compose. Its dense query uses cosine distance. BM25 is not in the database. Nothing in CI executes `PgVectorStore`.

## Bugs fixed

1. **Thousands without a comma were truncated.** `extract_money("$1500")` and `extract_money("$1500.00")` returned `150.00`. `extract_money("$10000")` returned `100.00`. The pattern tried a 1–3 digit group before a bare digit run, so it stopped after three digits when there was no comma. `MONEY_RE` now requires a comma group for the grouped form and otherwise takes the full digit run. The same pattern is used when a sum picks the amount nearest a fee name.

2. **A fee clause contributed its balance threshold, and a repeated sentence was summed twice.** On text that only had the fee clauses, `plan_tools("Calculate the total of the fees.")` returned `35.00`, `12.00`, and `1500.00`. The `$1,500.00` is the balance that triggers the maintenance fee. Two copies of the same sentences (what the 80-character chunk overlap can produce) returned `94.00`. Keyword sums now take one amount per sentence, the amount nearest the category word, and skip a sentence already counted.

3. **`sum_currency` displayed addends that did not add up to the total.** `sum_currency(["1.005", "1.005"])` returned amounts `1.00` and `1.00` with total `2.01`. Formatting used the default round-half-even on each addend, while the total rounded the raw sum half-up. Each addend is now quantized half-up to cents, then added. The same inputs return `1.01`, `1.01`, and `2.02`. `1.015 + 1.015` returns `1.02`, `1.02`, and `2.04`. `35.00 + 12.00` is still `47.00`.

4. **An unnamed or unrecognized total summed every dated posting.** `Calculate the total.`, `Calculate the total of the amounts.`, and `Calculate the total of the payments.` each returned `$6819.17` (eleven dated lines, deposits and withdrawals together). A question with no fee, deposit, withdrawal, or payment category now returns no tool call, and the graph refuses. `Calculate the total of the payments.` sums the dated lines that say payment (`1850.00` and `412.63`, total `2262.63`).

5. **Deposit and withdrawal totals ignored the section.** `Calculate the total of the deposits.` refused. The deposit lines do not contain the word "deposit", and the line `Total deposits:` is skipped so it is not added on top of the parts. `Calculate the total of the withdrawals.` only saw the ATM line, and a single amount is not enough for `plan_tools`, so that refused too. Lines in a section whose title starts with the category word are included. On the sample statement the deposit tool total is `3962.72` and the withdrawal tool total is `2856.45` (the withdrawal section includes the two fee postings, which is also what the summary line states).

6. **"How much altogether" was routed to compute and then refused.** `classify_question` treats that phrase as compute. `plan_tools` did not, so the compute node stored no tool result and refused. The phrase now both triggers a sum and, when the words after it name specific lines, uses the one-line-per-name matcher. `How much altogether are the overdraft fee and the ATM withdrawal?` totals `235.00` (`35.00` and `200.00`), not every fee.

7. **Question words hid sentences that are in the PDFs.** `when` and `where` were missing from `STOPWORDS`, next to `what` / `why` / `how`. `when` occurs in the maintenance-fee clause and in the poisoned notice (`When asked any question...`). For `When does the policy automatically renew?` two of the three query tokens were in the renewal sentence (printed overlap 0.67), under the 0.75 grade cutoff. The notice was quarantined, and the reply was the blocked-document refusal. `amount` was a query token that is not in `Total withdrawals: $2,856.45`, so `What is the total withdrawals amount?` had the same 0.67 overlap and was refused. `when` and `where` are stopwords. `amount` and `amounts` are query-only ignore words. Both questions now answer from the sample text (`automatically renew`, and `2,856.45`).

8. **The string `"false"` was treated as true, and a mixed-case route was dropped.** `bool("false")` is true, and `grade` / `answer` used `bool(...)` on model JSON. A hosted grader returning `{"relevant": "false"}` would have kept the chunk. An answerer returning `{"refused": "false"}` would have been refused. `route` compared the raw string to `answer|compute|risk|refuse`, so `" Compute "` became a fact question. `coerce_bool` handles booleans and those strings. The route is stripped and lowercased. `parse_json_object` decodes the first object only, so trailing prose or a second object no longer invalidates the first. A provider exception on the route node still refuses.

9. **Some SSN and account shapes were not masked.** `redact("SSN 999999999")` returned the digits. `redact("Account number: 4821-9377-01")` returned the dashed number. The label list had no `ssn` abbreviation, and the digit run could not contain a space or hyphen. Both are masked now (`[REDACTED:SSN]`, `****7701`). `contains_raw_pii` uses the same labeled pattern, so a value `redact` would mask counts as raw PII before redaction. The sample account `4821937701` and the dashed sample SSN `999-99-9999` still mask as before.

10. **A sentence that matched two clause types kept only the first.** `scan_risks` broke out of the pattern list after one hit, and broke again when that same sentence and type had already been seen, which skipped later types. A sentence with a late fee and an automatic renewal produced only `auto_renewal`. It now records each matching type once per sentence and page. The sample corpus has no sentence that matches two patterns, so the golden risk question still returns the same clause families.

11. **The upload limit was applied after the whole body was read, and the error text ignored the setting.** `POST /documents` called `file.read()` with no size, then `ingest_pdf` compared the length to `max_upload_bytes` and always said `10 MB`. The handler now reads `limit + 1` bytes. The error reports the configured limit. A 32-byte limit rejects a longer `%PDF` body with HTTP 400 and does not add a document.

## Tests added

`pytest` collected 33 tests after these changes (18 before). Each bug above has a regression in this list. The eval script was re-run and still printed 1.000 on every metric, above the floors, and exited 0.

| Test | Why it is there |
| --- | --- |
| `test_extract_money_keeps_thousands_without_commas` | `$1500`, `$1500.00`, and `$10000` must keep every digit. Comma forms and `$42.17` stay the same. |
| `test_uncommaed_thousands_are_summed_in_full` | A named sum of `$1500.00` and `$20.00` is `1520.00`, so the tool path uses the fixed regex. |
| `test_fee_total_ignores_balance_threshold_and_duplicate_lines` | Clause-only fee text sums to `47.00`, not `1547.00`. A duplicated chunk does not double the total. |
| `test_sum_currency_rounds_each_addend_half_up` | `1.005 + 1.005` and `1.015 + 1.015` display addends that equal the total. |
| `test_unnamed_total_does_not_sum_the_whole_statement` | `Calculate the total.` and `Calculate the total of the amounts.` do not call a tool. |
| `test_category_totals_use_the_posted_lines` | Sample deposits `3962.72`, withdrawals `2856.45`, payments `2262.63`, and the "how much altogether" fee and ATM cases. |
| `test_question_words_do_not_hide_a_sentence_that_is_in_the_document` | The renewal question and the total-withdrawals question answer from the PDFs and stay grounded. |
| `test_model_json_strings_do_not_flip_booleans_or_drop_the_object` | String `false`, route ` Compute `, a provider exception, a trailing second JSON object, and an answerer that raises. |
| `test_redacts_undashed_ssn_and_separated_account_number` | `SSN 999999999`, `4821-9377-01`, and a spaced account. `contains_raw_pii` agrees before and after redaction. |
| `test_one_sentence_can_flag_more_than_one_clause_type` | Late fee plus auto-renewal yields both types, including when the chunk is passed twice. |
| `test_oversized_pdf_is_rejected` | A body past a 32-byte `max_upload_bytes` is HTTP 400 and is not indexed. |
| `test_ask_is_limited_to_the_requested_documents` | `doc_ids` pointing at the policy refuses the bank balance and still answers the flood question. |
| `test_blank_question_and_reingest` | A blank question raises `ValueError`. Ingesting a sample PDF again does not add a fifth document. |
| `test_amortized_payment_rejects_a_non_positive_term` | `months=0` raises `ValueError`. |
| `test_extract_pages_rejects_bytes_that_are_not_a_pdf` | Non-PDF bytes raise `PdfParseError`. |

## Simplifications

- `sum_currency` no longer builds an unused `parsed` list and then formats the amounts a second way.
- `grounding_score` had a second branch that required a token to be both in and not in `context_numbers`. It was removed. Behavior is unchanged.
- `risk._sentences` only forwarded to `textutil.sentences`. Callers use `sentences` directly.
- `_llm_answer` imported `NOT_FOUND` inside the function even though the module already imported it.
- Category detection for sums is `_keyword_for` / `_section_matches` instead of a nested copy of the old keyword loop.
- Hosted JSON booleans go through `coerce_bool` instead of `bool`.

## Not done

- `Settings.retrieval_k` defaults to 4 and is never read. `retrieve` hardcodes `k=4`, or `k=6` when the route is `risk`.
- `ingest_pdf` sets `synthetic=True` for every file, including uploads. `PgVectorStore` does not store the flag; rows read back from `documents` also default to true.
- `PgVectorStore` has no test. CI does not start Postgres.
- Injection coverage is the list in `INJECTION_PATTERNS`. A paraphrase of those phrases is not detected.
- Log redaction does not walk nested dicts or lists.
- `Sum the overdraft fee and the maintenance fee` (no `sum of`) is classified as `answer`, not `compute`. The extractive path copies sentences; it does not add them.
- `payment` is matched as a substring, so a future question that only says `prepayment` can fall into the payment category if it is not handled as a named line.
- The lexical rerank is not a cross-encoder. There is no OCR and no table parser. Chunks are the text pypdf returns.
- `sum_currency([])` returns total `0.00`. Callers require at least two amounts before they invoke it.
- Compose starts Streamlit after the API container is created, not after `/health` succeeds.
- Grounding's template-word list is hand maintained. A hosted answer that uses other glue words can fail the 0.72 overlap check even when the numbers match.
- The hash embedder can map two tokens to the same bucket. That is acceptable for this corpus only in the sense that the golden file passes with it; it is not a general embedding model.
