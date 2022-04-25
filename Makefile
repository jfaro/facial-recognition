VENV = env
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

create-env: 
	python3 -m venv $(VENV)

activate-env:
	source $(VENV)/bin/activate

install: requirements.txt
	$(PIP) install -r requirements.txt

run: $(VENV)/bin/activate
	flask run

clean:
	rm -rf __pycache__

.PHONY: activate install run clean