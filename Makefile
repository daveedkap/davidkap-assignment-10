# Virtual environment directory
VENV = .venv

# Python interpreter to use
PYTHON = python3

# Create and set up the virtual environment, and install required dependencies
install:
	$(PYTHON) -m venv $(VENV) # Create a virtual environment in the specified directory
	. $(VENV)/bin/activate && pip install -r requirements.txt # Activate the virtual environment and install dependencies

# Run the Flask application
run:
	. $(VENV)/bin/activate && python -m flask --app app.py --debug run --host=0.0.0.0 --port=8000 # Activate the virtual environment and start the Flask development server

# Clean up by removing the virtual environment directory
clean:
	rm -rf $(VENV) # Delete the virtual environment directory entirely
