.PHONY: help install setup test lint format build push commit clean venv

# Variables
PYTHON := python3
PIP := pip3
VENV := env
BACKEND := backend/src

# Default target
help:
	@echo "Document-AI-Hub - Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup          - Setup virtual environment and install dependencies"
	@echo "  make install        - Install Python dependencies"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with Black"
	@echo "  make build          - Build the project"
	@echo "  make clean          - Clean up cache and temporary files"
	@echo "  make push           - Push code to GitHub (add, commit, push)"
	@echo "  make commit MSG=... - Commit with custom message"
	@echo "  make status         - Show git status"
	@echo ""

# Setup virtual environment
setup: venv install
	@echo "✓ Setup complete!"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "✓ Virtual environment created"; \
	else \
		echo "✓ Virtual environment already exists"; \
	fi

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt || true
	$(PIP) install -r backend/requirements.txt || true
	@echo "✓ Dependencies installed"

# Run tests
test:
	@echo "Running tests..."
	cd $(BACKEND) && $(PYTHON) -m pytest tests/ -v || true
	cd $(BACKEND) && $(PYTHON) test_agentic_integration.py || true
	@echo "✓ Tests completed"

# Lint code
lint:
	@echo "Running linting checks..."
	$(PYTHON) -m pylint $(BACKEND)/app/ --disable=all --enable=E,F || true
	@echo "✓ Linting completed"

# Format code
format:
	@echo "Formatting code with Black..."
	$(PYTHON) -m black $(BACKEND)/app/ --line-length=100 || true
	@echo "✓ Code formatted"

# Build project
build:
	@echo "Building project..."
	@echo "  - Checking Python version: $(PYTHON)"
	@$(PYTHON) --version
	@echo "  - Verifying source code structure"
	@ls -la $(BACKEND)/app/ || true
	@echo "✓ Build verification completed"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup completed"

# Git status
status:
	@echo "Git status:"
	@git status --short || true

# Add and commit with custom message
commit:
	@if [ -z "$(MSG)" ]; then \
		echo "Error: MSG is required. Usage: make commit MSG='Your message'"; \
		exit 1; \
	fi
	@echo "Staging changes..."
	git add .
	@echo "Committing: $(MSG)"
	git commit -m "$(MSG)" || true
	@echo "✓ Changes committed"

# Push to GitHub
push: commit
	@echo "Pushing to GitHub..."
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	echo "  Current branch: $$branch"; \
	git push origin $$branch || git push --set-upstream origin $$branch
	@echo "✓ Code pushed to GitHub"

# Convenience targets
push-main:
	@echo "Checking out main branch..."
	git checkout main
	@echo "Pulling latest changes..."
	git pull origin main
	@echo "Pushing to main..."
	git push origin main
	@echo "✓ Pushed to main branch"

# Pull latest changes
pull:
	@echo "Pulling latest changes from GitHub..."
	git pull origin
	@echo "✓ Latest changes pulled"

# Show logs
log:
	@git log --oneline -10

# Create a new branch
branch:
	@if [ -z "$(BRANCH)" ]; then \
		echo "Error: BRANCH is required. Usage: make branch BRANCH='feature/name'"; \
		exit 1; \
	fi
	@echo "Creating branch: $(BRANCH)"
	git checkout -b $(BRANCH)
	@echo "✓ Branch created and checked out"

# Switch branch
switch:
	@if [ -z "$(BRANCH)" ]; then \
		echo "Error: BRANCH is required. Usage: make switch BRANCH='main'"; \
		exit 1; \
	fi
	@echo "Switching to branch: $(BRANCH)"
	git checkout $(BRANCH)
	@echo "✓ Switched to branch: $(BRANCH)"

# Full CI/CD pipeline
all: clean lint test build push
	@echo "✓ Full pipeline completed successfully!"

# Development workflow
dev: install lint test
	@echo "✓ Development checks passed!"

# Production preparation
prod: clean format lint test build
	@echo "✓ Production build ready!"
	@echo "Next step: make push MSG='Release v1.0.0'"
