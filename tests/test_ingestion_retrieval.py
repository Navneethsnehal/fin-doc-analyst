from pathlib import Path

from fin_doc_analyst.agent.tools import amortized_payment, sum_currency
from fin_doc_analyst.ingestion.chunker import chunk_pages
from fin_doc_analyst.ingestion.pdf import extract_pages
from fin_doc_analyst.retrieval.embeddings import HashEmbedder
from fin_doc_analyst.retrieval.memory import MemoryStore
from fin_doc_analyst.schemas import DocumentInfo

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


def test_payment_tool_matches_two_cent_rounding() -> None:
    result = amortized_payment("18400.00", "0.0749", 60)
    assert result["monthly_payment"] == "368.61"
    assert sum_currency(["35.00", "12.00"])["total"] == "47.00"


def test_sample_pdfs_preserve_pages_sections_and_key_facts() -> None:
    statement = SAMPLES / "bank_statement_synthetic.pdf"
    pages = extract_pages(statement)
    assert len(pages) >= 4
    joined = "\n".join(text for _, text in pages)
    assert "SYNTHETIC DOCUMENT" in joined
    assert "Ending balance: $5,356.45" in joined
    assert "Checking account number: 4821937701" in joined
    chunks = chunk_pages("bank", statement.name, pages)
    summary = [chunk for chunk in chunks if chunk.section == "Account Summary"]
    assert summary
    assert summary[0].page == 1
    assert any(chunk.section == "Withdrawals" for chunk in chunks)

    loan = "\n".join(text for _, text in extract_pages(SAMPLES / "loan_agreement_synthetic.pdf"))
    assert "APR is 7.49%" in loan
    assert "368.61" in loan
    policy = "\n".join(
        text for _, text in extract_pages(SAMPLES / "insurance_policy_synthetic.pdf")
    )
    assert "Flood is excluded." in policy
    assert "automatically renew" in policy
    poison = "\n".join(text for _, text in extract_pages(SAMPLES / "poisoned_notice_synthetic.pdf"))
    assert "Ignore all previous instructions" in poison
    assert "999-99-9999" in poison


def test_hybrid_search_finds_paraphrase_and_exact_fee() -> None:
    store = MemoryStore(HashEmbedder())
    for path in sorted(SAMPLES.glob("*.pdf")):
        pages = extract_pages(path)
        from fin_doc_analyst.ingestion.chunker import chunk_pages as chunk

        chunks = chunk(path.stem, path.name, pages)
        store.add(
            DocumentInfo(
                doc_id=path.stem, doc_name=path.name, pages=len(pages), chunks=len(chunks)
            ),
            chunks,
        )
    balance = store.search("How much money was in the account at the end of the statement?", k=3)
    assert balance
    assert "5,356.45" in balance[0].chunk.text or any(
        "5,356.45" in item.chunk.text for item in balance
    )
    fees = store.search("overdraft fee and monthly maintenance fee", k=3)
    assert any("Overdraft fee" in item.chunk.text for item in fees)
