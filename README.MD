# Deepseek-v2 Advanced Prompting Techniques Project

## Overview
This repository contains code and experimental setups for applying advanced prompting techniques to improve the performance of the deepseek-v2 model on the  MBPP datasets. The main focus is on comparing Chain of Thought prompting (CoT), In-context Learning (ICL), and Reflexion against naïve prompting techniques.

## Repository Structure

- `/NLP_RAG`
  - `/dataset`
    - `/gsm8k` - Contains the GSM8k dataset used for experiments.
    - `/mbpp` - Contains the MBPP dataset used for experiments.
- `/log` - Contains log files generated during the execution of experiments.
- `/utils`
  - `api_utils.py` - Utility functions for interacting with the Deepseek-v2 API.
  - `file_utils.py` - Functions for file manipulation and storage handling.
  - `mp_utils.py` - Utilities for multiprocessing and data handling.
- `DataHelper.py` - Helper functions for data loading and examining.
- `mbpp.py` - Script for testing MBPP dataset .
- `my_method.py` - Implementation of custom prompting methods.
- `rag_prepare.py` - Preparation scripts for RAG (Retrieval-Augmented Generation)
- `.gitignore` - Specifies intentionally untracked files to ignore.
- `README.MD` - Describes the project, its structure, and how to run the scripts.
- `requirements.txt` - Lists all the dependencies necessary to run the project.

## Setup and Installation

1. Clone the repository:
`!pip install -r requirements`
2. Add the api_key:
`!export DEEPSEEK_API_KEY="XXXXXX"`
3. run the code:
`!python mbpp.py`