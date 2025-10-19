# mmGraphRAG ODSC West 2025
This repo is a companion to the mmGraphRAG presentation ODSC West 2025. Multimodal GraphRAG (mmGraphRAG) is the natural extension of GraphRAG which incorporates visual information into graph based retrieval.

__Contact Information__
- Author, Presenter [David Hughes](mailto:david@hyperdimensionalcomputing.ai) 
- Co-Presenter [Amy Hodler](mailto:amy.hodler@graphgeeks.org)

## Core Requirements
- Python3.10
- Poetry ->  Package management **This will be migrated to uv**
  - [installation instructions](https://python-poetry.org/docs/)
- Ollama -> Local language model hosting
  - [installation instructions](https://ollama.com/download)

## Usage
1. Install all core requirements listed above
2. Pull the vision model [LLaVA](https://ollama.com/library/llava): Large Language and Vision Assistant
    `ollama pull llava`
3. Clone the repo for this project
   `git clone {{repo_path}}`
4. Change directory to the new project directory. For example, if you cloned to your dev directory:
   `cd ~/dev/mmgrgraphrag`
5. Start a Poetry shell. This will create a virtual environment and install any project dependencies. If you want to see what packages will be installed before running this command you can open the file _pyproject.toml_ to see what will be installed.
   `poetry shell`
6. Run `poetry show` to see installed dependencies. You may need to run `poetry install`.
7. Change to the notebook directory
    `cd src/mmgraphrag_odsc_West_2025/notebook/`
8. Initialize BAML
   `baml-cli init`
9.  Generate the BAML client
    `baml-cli generate`
10. You should now have a directory structure that looks like this:
    ```
    .
    ├── README.md
    ├── poetry.lock
    ├── pyproject.toml
    └── src
        └── mmgraphrag_odsc_West_2025
            ├── __init__.py
            ├── baml_client
            │   ├── __init__.py
            │   ├── async_client.py
            │   ├── globals.py
            │   ├── inlinedbaml.py
            │   ├── partial_types.py
            │   ├── sync_client.py
            │   ├── tracing.py
            │   ├── type_builder.py
            │   └── types.py
            ├── baml_src
            │   ├── captions.baml
            │   ├── clients.baml
            │   ├── features.baml
            │   └── generators.baml
            ├── config
            │   ├── __init__.py
            │   └── config.py
            ├── images
            │   ├── b_COCO_val2014_000000000715.jpg
            │   ├── b_COCO_val2014_000000001153.jpg
            │   ├── b_COCO_val2014_000000002587.jpg
            │   ├── b_COCO_val2014_000000003661.jpg
            │   ├── b_COCO_val2014_000000003817.jpg
            │   ├── e_COCO_val2014_000000000757.jpg
            │   ├── e_COCO_val2014_000000001164.jpg
            │   ├── e_COCO_val2014_000000001869.jpg
            │   ├── e_COCO_val2014_000000002255.jpg
            │   └── e_COCO_val2014_000000004229.jpg
            ├── lib
            │   ├── __init__.py
            │   ├── embedding.py
            │   └── image_processing.py
            ├── notebook
            │   └── mmgraphrag.ipynb
            ├── tensorboard
            │   ├── checkpoint
            │   ├── metadata.tsv
            │   ├── model.ckpt.data-00000-of-00001
            │   ├── model.ckpt.index
            │   ├── model.ckpt.meta
            │   ├── projector_config.pbtxt
            │   ├── sprite.png
            │   └── vectors.tsv
            └── utils
                ├── __init__.py
                └── logger.py

    17 directories, 60 files
    ```
1.  Start jupyter lab
   `jupyter lab`
2.  Jupyter Lab should open a browser and present its UI with the notebook named _mmgraphrag.ipynb_ being visibile on the left panel. If not, open a browser and navigate to `localhost:8888/lab/`
3.  Follow the instructions in the notebook
