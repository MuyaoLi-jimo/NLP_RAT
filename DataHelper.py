"""
preparing the dataset and evaluation matrix
@author muyao
"""

from datasets import load_dataset,load_from_disk,DatasetDict,Dataset
from pathlib import Path
from utils import file_utils

class DatasetLoader:
    def __init__(self):
        self.datasets = {}
        self.available_dataset = {"gsm8k","mbpp","gaokao_obj"}
        
    def get_dataset(self,dataset_name:str):
        """ 
        获取dataset格式的数据集
        """
        assert dataset_name in self.available_dataset
        
        dataset = self.datasets.get(dataset_name)
        # 如果已经在datasets中，直接提取
        if dataset:
            return dataset
        
        if dataset_name == "gsm8k":
            dataset = self.get_gsm8k()
        elif dataset_name == "mbpp":
            dataset = self.get_mbpp()
        elif dataset_name == "gaokao_obj":
            dataset = self.get_gaokao(is_obj=True)
        else:
            raise AssertionError(f"do not define dataset: {dataset_name}")
        
        self.datasets[dataset_name] = dataset
        return dataset
                     
    def get_gaokao(self,is_obj=True)->DatasetDict:
        # 如果已经在本地保存
        dataset_name = "gaokao_obj" if is_obj else "gaokao_sub"
        dataset_path = Path(__file__).parent/"dataset"/dataset_name        
        if dataset_path.exists():
            dataset_index_path = dataset_path /"dataset_dict.json"
            dataset_index = file_utils.load_json_file(dataset_index_path)["splits"]
            dataset = {}
            for subset_name in dataset_index:
                dataset_subset_path = dataset_path / subset_name
                dataset[subset_name] = load_from_disk(dataset_subset_path)
            dataset = DatasetDict(dataset)
            return dataset
        ## 下面是第一次访问时的处理方式
        
        index_file_name = "Obj_Prompt.json" if is_obj else "Sub_Prompt.json"
        gaokao_raw_dataset =  Path(__file__).parent/"GAOKAO-Bench"/"Bench"/ index_file_name
        index_data = file_utils.load_json_file(gaokao_raw_dataset)["examples"]
        dataset_dict = {}
        for i in range(len(index_data)):
            question_type = index_data[i]['type']
            zero_shot_prompt_text = index_data[i]['prefix_prompt']
            keyword = index_data[i]['keyword']
            subset_name = f"{keyword}-{question_type}"
            dataset_fold_type = "Objective_Questions" if is_obj else "Subjective_Questions"
            subset_path = Path(__file__).parent/"GAOKAO-Bench"/"Data"/dataset_fold_type/f"{keyword}.json"
            subset_data = file_utils.load_json_file(subset_path)["example"]
            sub_ds = Dataset.from_list(subset_data)
            def preprocess(example):
                example["id"] = keyword + str(example["index"])
                del example["index"]
                return example
            sub_ds = sub_ds.map(preprocess,
                desc="polish index",
            )
            # get 10 items as examples
            sub_dataset_dict = sub_ds.train_test_split(test_size=10)
            rename_sub_dataset_dict = DatasetDict({
                "test" : sub_dataset_dict["train"],
                "prompt" : sub_dataset_dict["test"],
            })
            dataset_dict[subset_name] = rename_sub_dataset_dict
        dataset = DatasetDict(dataset_dict)
        dataset.save_to_disk(dataset_path)
        return dataset
     
    def get_mbpp(self)->DatasetDict:
        dataset_path = Path(__file__).parent/"dataset"/"mbpp"
        if dataset_path.exists():
            print(dataset_path)
            return load_from_disk(dataset_path)
        # load from hf
        dataset = load_dataset("mbpp","sanitized")
        dataset = dataset.rename_column("task_id","id")
        pattern = r'(def .+?\)) *: *'
        import re
        def preprocess(example):
            code = example["code"]
            
            match = re.search(pattern, code)
            if match:
                # 提取定义以及定义之前的所有内容
                example["def"]= match.group(1)
                return example
            else:
                print(example)
                raise AssertionError("no way")
        dataset = dataset.map(preprocess,
            desc="getting final answer",
        )
        dataset.save_to_disk(dataset_path)
        return dataset
    
    def get_gsm8k(self):
        dataset_path = Path(__file__).parent/"dataset"/"gsm8k"
        if dataset_path.exists():
            return load_from_disk(dataset_path)
        # load from hf
        dataset = load_dataset("openai/gsm8k","main")
        # 预处理，提取答案
        import re
        import uuid
        pattern = r"####\s*(-?\d+)"  #'#### 数字'
        def preprocess(example):
            match = re.search(pattern, example["answer"])
            if match:
                example["gt"] = int(match.group(1))
            else:
                print(example)
                raise AssertionError("no way")
            example["id"] = str(uuid.uuid4())
            return example
        dataset = dataset.map(preprocess,
            desc="getting final answer",
        )
        test_prompt_dataset = dataset["test"].train_test_split(test_size=5)
        dataset = DatasetDict({
            "train":dataset["train"],
            "test":test_prompt_dataset["train"],
            "prompt":test_prompt_dataset["test"],
        })
        dataset.save_to_disk(dataset_path)
        return dataset


if __name__ == "__main__":
    data_loader = DatasetLoader()
    dataset = data_loader.get_dataset("gaokao_obj")
    item = dataset["2010-2022_Math_II_MCQs-single_choice"]["test"][0]
    print(item)

    