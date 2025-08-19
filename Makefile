# Makefile for FastAPI project

.PHONY: dev run logs

# Start development server with auto-reload
dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run server in production mode (no auto-reload)
run:
	python3 app/main.py

# View log output
logs:
	tail -f log/model.log
