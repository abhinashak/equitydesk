# Streamlit UI
streamlit run ui/app.py

# CLI / standalone
python -m cli.main --help
python -m cli.main ticker historical --check-rebase
python -m cli.main fundamental run --tickers BEL BDL
python -m cli.main signal generate
python -m cli.main trade plan --capital 2000000 --phases 3
python -m cli.main config set MOCK_MODE false

# Tests
pytest tests/
