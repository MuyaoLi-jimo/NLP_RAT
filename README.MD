# NLP_RAG


## Repository Structure
- `/src`
  - `/dataset`
    - `collect_data.py` - collect data api
    - `DataHelper.py`  
  - `/evaluate`
    - `evaluate.py`  - evaluate api
  - `/Method`
    - `Method.py`    - function api
    - `rag.py`       - rag system
    - `rat_tools.py` - the tool for RAT
  - `/utils`
    - `api_utils.py`  - Utility functions for interacting with the Deepseek-v2 API.
    - `file_utils.py` - Functions for file manipulation and storage handling.
    - `mp_utils.py`   - Utilities for multiprocessing and data handling.
  - `/visial`
  + `gaokao_obj.py`   - the main tunnel for gaokao benchmark
  + `mbpp.py`
- `/examples`
  - `rat_tools.py`
  - `techniques.py`
  - `test.ipynb`
- `/docs`
  - `proposal.md`
- `/lib`
  - `/GAOKAO-Bench`
- `/logs`
- `.gitignore` - Specifies intentionally untracked files to ignore.
- `README.MD` - Describes the project, its structure, and how to run the scripts.
- `requirements.txt` - Lists all the dependencies necessary to run the project.

## Setup and Installation

1. Clone the repository:
`$ pip install -r requirements`
2. Add the api_key:
`$ export DEEPSEEK_API_KEY="XXXXXX"`
3. run the code:
`$ python gaokao_obj.py --model_name DeepSeek-V3`

## rag resource
- chemistry
  - [https://github.com/Seeridia/Chemistry-Note](https://github.com/Seeridia/Chemistry-Note)
  - [https://www.hxzxs.cn/view-20008-1.html](https://www.hxzxs.cn/view-20008-1.html)
- physics
  - [https://github.com/asjdf/HighSchoolPhysicsReview](https://github.com/asjdf/HighSchoolPhysicsReview)
  - [https://github.com/Campanulata/High-school-physical-data](https://github.com/Campanulata/High-school-physical-data)
  - [https://github.com/Ayvytr/KnowledgeHierarchy/blob/master/%E9%AB%98%E4%B8%AD%E7%89%A9%E7%90%86.md?plain=1](https://github.com/Ayvytr/KnowledgeHierarchy/blob/master/%E9%AB%98%E4%B8%AD%E7%89%A9%E7%90%86.md?plain=1)
- geography
  - [https://github.com/Cicada000/Geography-Notes](https://github.com/Cicada000/Geography-Notes)
  - [https://github.com/a15355447898a/Geography_Note/tree/main](https://github.com/a15355447898a/Geography_Note/tree/main)
  