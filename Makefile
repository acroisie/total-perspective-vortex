.PHONY: all clean fclean re

all: .venv

.venv: requirements.txt
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

clean:
	rm -rf __pycache__ */__pycache__ *.pyc .mypy_cache

fclean: clean
	rm -rf .venv

re: fclean all
