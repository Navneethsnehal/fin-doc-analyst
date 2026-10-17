.PHONY: install lint test eval demo api

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest

eval:
	python scripts/run_eval.py

api:
	uvicorn fin_doc_analyst.api.app:app --reload

demo:
	streamlit run apps/streamlit_app.py
