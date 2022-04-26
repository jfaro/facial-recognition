VENV = env
PIP = $(VENV)/bin/pip

install: requirements.txt
	$(PIP) install -r requirements.txt

run: $(VENV)/bin/activate
	flask run

clean:
	rm -rf __pycache__

.PHONY: install run clean