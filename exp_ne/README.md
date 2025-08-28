# Named-Entity-copying experiment

This directory contains the code used to run the named-entity-copying experiments.
The directory consists of the following files:
- `eval_dummy.json`: The evaluation results based on the dummy sentences with the copied NEs.
- `flores_ner.py`: The code used to extract named entities with GPT-4o-mini. Note that, for the API request, you need to set your OpenAI API key in a separate `.env` file, which will be loaded by `dotenv` (see also their documentation: https://pypi.org/project/python-dotenv/). Alternatively, you can also set the environment variable through the shell, e.g., `export OPENAI_API_KEY=<your_api_key>`.
- `pyproject.toml`: This file contains the requirements of the environment to run the experiments. I use Poetry (https://python-poetry.org/) to manage the dependency. If you have Poetry installed, you can build a virtual environment and install required packages by simply running `poetry install`.
- `results.json`: This JSON file contains the list of the sentences after extracting only named entities from the FLORES+ English sentences. This is the generated result file after running `flores_ner.py`.
- `system_prompt.txt`: This is the system prompt text used in `flores_ner.py`.
- `user_prompt_template.txt`: This is the user prompt template text used in `flores_ner.py`.
- `visualize.ipynb`: This is the code snippets (Notebook) to visualize the results.