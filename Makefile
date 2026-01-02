.PHONY: venv install demo demo_pklemon demo_sance report clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

demo:
	. .venv/bin/activate && python scripts/run_demo.py --config configs/demo.yaml

demo_pklemon:
	. .venv/bin/activate && python scripts/run_demo.py --config configs/pklemon_public.yaml

demo_sance:
	. .venv/bin/activate && python scripts/run_demo.py --config configs/sance_public.yaml

report:
	. .venv/bin/activate && python -m src.reporting.summary

clean:
	rm -rf outputs
