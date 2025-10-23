# Justfile for mmGraphRAG ODSC West 2025 Project
# This file provides convenient commands for managing the project

# Default recipe - show available commands
default:
    @just --list

# Generate BAML client code
generate-baml:
    @echo "Generating BAML client code..."
    uv run baml-cli generate --from src/mmgraphrag_odsc_west_2025/baml_src

# Populate the LanceDB database with examples for dICL system
populate-db:
    @echo "Populating LanceDB database with dICL examples..."
    @echo "This will generate examples for bananas, technology, and functional programming topics."
    @echo "Make sure Ollama is running with the phi4 model before proceeding."
    @echo ""
    uv run populate_dicl.py

# Initialize existing LanceDB database (for testing queries)
init-db:
    @echo "Initializing connection to existing LanceDB database..."
    uv run python3 -c "from src.mmgraphrag_odsc_west_2025.lib.dicl_system import DICLSystem; system = DICLSystem(); system.initialize_database('dicl_examples'); print('Database initialized successfully!')"

# Test the dICL system with sample queries
test-dicl:
    @echo "Testing dICL system with sample queries..."
    uv run python3 -c "from src.mmgraphrag_odsc_west_2025.lib.dicl_system import DICLSystem; system = DICLSystem(); system.initialize_database('dicl_examples'); print('Testing banana query...'); result = system.process_query('What are the health benefits of bananas?'); print(f'Answer: {result[\"answer\"]}'); print(f'Found {len(result[\"examples\"])} similar examples'); print('\nTesting lambda calculus query...'); result = system.process_query('What is lambda calculus?'); print(f'Answer: {result[\"answer\"]}'); print(f'Found {len(result[\"examples\"])} similar examples')"

# Clean up generated files and databases
clean:
    @echo "Cleaning up generated files..."
    rm -rf dicl_examples/
    rm -rf src/mmgraphrag_odsc_west_2025/__pycache__/
    rm -rf src/mmgraphrag_odsc_west_2025/**/__pycache__/
    rm -rf src/mmgraphrag_odsc_west_2025/baml_client/__pycache__/
    @echo "Cleanup complete!"

# Install project dependencies
install:
    @echo "Installing project dependencies..."
    uv sync

# Run Jupyter notebook
notebook:
    @echo "Starting Jupyter notebook..."
    uv run jupyter lab

# Full setup: install dependencies, generate BAML, and populate database
setup: install generate-baml populate-db
    @echo "Setup complete! Database populated with examples."

# Help command
help:
    @echo "Available commands:"
    @echo "  populate-db    - Generate and populate LanceDB with dICL examples"
    @echo "  populate-db-inline - Alternative populate method using inline Python"
    @echo "  init-db        - Initialize connection to existing database"
    @echo "  test-dicl      - Test the dICL system with sample queries"
    @echo "  generate-baml  - Generate BAML client code"
    @echo "  clean          - Clean up generated files and databases"
    @echo "  install        - Install project dependencies"
    @echo "  notebook       - Start Jupyter notebook"
    @echo "  setup          - Full setup (install + generate + populate)"
    @echo "  help           - Show this help message"
