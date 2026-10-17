PY ?= python3
CACHE := $(CURDIR)/cache/llm_cache.sqlite

.PHONY: help setup unpack data tables figures rq4 replay verify-replay clean

help:
	@echo "make setup   - create .venv with the pinned analysis environment"
	@echo "make unpack  - decompress the cache and large files, verify checksums"
	@echo "make data    - rebuild the CleanVul pairs from the public release (needs network)"
	@echo "make tables  - regenerate all paper tables from results/ and compare with the paper (~1 min)"
	@echo "make figures - regenerate the result figures from results/"
	@echo "make rq4     - recompute RQ4 agreement and tables from the coding sheets"
	@echo "make replay  - rerun the whole pipeline from the frozen cache, no model calls (~2 h CPU; needs srcML + srcSlice)"

setup:
	$(PY) -m venv .venv && .venv/bin/pip install -r environment/requirements.txt

unpack:
	$(PY) scripts/unpack.py

data: unpack
	$(PY) data/fetch_cleanvul.py

tables:
	mkdir -p out
	$(PY) paper/paper_tables_df89e.py results out/tables.md
	diff -u paper/expected_tables.md out/tables.md && echo "tables: identical to the paper"

figures:
	mkdir -p out/figures
	$(PY) paper/paper_figures_df89e.py results rq4 out/figures

rq4:
	rm -rf out/rq4 && mkdir -p out && cp -R rq4 out/rq4
	$(PY) rq4/scripts/rq4_agreement.py out/rq4 && $(PY) rq4/scripts/rq4_tables.py out/rq4
	diff rq4/df89e_rq4_agreement.json out/rq4/df89e_rq4_agreement.json && echo "rq4 agreement: identical"

replay: data
	rm -rf out/replay && mkdir -p out/replay && cp src/*.py out/replay/ && cp -R data out/replay/data
	cd out/replay && $(PY) fetch_data.py && RP_LLM_CACHE=$(CACHE) $(PY) main.py --full
	cd out/replay && $(PY) loc_chance.py results && $(PY) loc_lift_test.py results && $(PY) loc_tests.py results
	$(MAKE) verify-replay

verify-replay:
	$(PY) scripts/compare_results.py out/replay/results/results.json results/results.json
	$(PY) scripts/compare_results.py out/replay/results/loc_lift_test.json results/loc_lift_test.json

clean:
	rm -rf out
