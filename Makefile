.PHONY: venv render preview list segment clean clean-all

PYTHON := .venv/bin/python

venv: ## Create virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

render: venv ## Full assembly (~depends on segment count)
	$(PYTHON) assemble.py

preview: venv ## Preview mode (~15s per segment)
	$(PYTHON) assemble.py --preview

list: venv ## List segments and exit
	$(PYTHON) assemble.py --list

segment: venv ## Render one segment: make segment N=3
	$(PYTHON) assemble.py --segment $(N)

clean: ## Remove output/
	rm -rf output/

clean-all: clean ## Remove output/ + .venv/
	rm -rf .venv/
