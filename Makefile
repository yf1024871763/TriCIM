PYTHON ?= python3
PIP ?= pip3
CONFIG ?= configs/default.yaml
ACCELERGY_REPO ?= accelergy-timeloop-infrastructure
ACCELERGY_URL ?= https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git

.PHONY: install deps booksim run check clean

install:
	if [ ! -d "$(ACCELERGY_REPO)" ]; then \
		git clone --recurse-submodules "$(ACCELERGY_URL)" "$(ACCELERGY_REPO)"; \
	fi
	$(PIP) install -r requirements.txt
	$(MAKE) -C "$(ACCELERGY_REPO)" install_accelergy
	$(PIP) install "./$(ACCELERGY_REPO)/src/timeloopfe"
	$(MAKE) -C "$(ACCELERGY_REPO)" install_timeloop
	$(MAKE) -C booksim2/src

deps:
	$(PIP) install -r requirements.txt

booksim:
	$(MAKE) -C booksim2/src

run:
	$(PYTHON) main.py --config $(CONFIG)

check:
	$(PYTHON) -m py_compile main.py $$(find src -path '*/__pycache__' -prune -o -name '*.py' -print)

clean:
	find . -path '*/__pycache__' -type d -prune -exec rm -rf {} +
