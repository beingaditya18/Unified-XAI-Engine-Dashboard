.PHONY: install train run-ui run-api lint format test docker-build clean help

help:
	@echo "======================================================================"
	@echo "                     Unified XAI Engine Command Menu"
	@echo "======================================================================"
	@echo "install      : Install dependencies"
	@echo "train        : Run training pipeline and serialize models"
	@echo "run-ui       : Run Streamlit dashboard"
	@echo "run-api      : Run FastAPI API backend"
	@echo "lint         : Run ruff check and mypy static type checking"
	@echo "format       : Run black code formatter"
	@echo "test         : Run pytest test suite"
	@echo "docker-build : Build Docker images using Docker Compose"
	@echo "clean        : Remove caches and build artifacts"
	@echo "======================================================================"

install:
	pip install -r requirements.txt
	pre-commit install

train:
	python models/train_models.py

run-ui:
	streamlit run dashboard/app.py

run-api:
	uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload

lint:
	ruff check src/ explainability/ dashboard/ models/ tests/
	mypy src/ explainability/ models/

format:
	black src/ explainability/ dashboard/ models/ tests/

test:
	pytest tests/ -v --color=yes

docker-build:
	docker compose build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
