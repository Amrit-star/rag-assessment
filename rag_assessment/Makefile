.PHONY: install test benchmark app lint

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

benchmark:
	python run_benchmark.py

app:
	PYTHONPATH=. streamlit run app/Home.py

lint:
	python -m py_compile src/interfaces.py src/config.py src/embeddings.py src/vector_store.py src/query_expansion.py src/orchestrator.py run_benchmark.py
	@echo "Syntax OK"
