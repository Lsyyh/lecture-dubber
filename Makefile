.PHONY: test doctor

test:
	PYTHONPATH=src pytest -q

doctor:
	PYTHONPATH=src python -c 'from lecture_dubber.cli import app; print("CLI import OK")'
