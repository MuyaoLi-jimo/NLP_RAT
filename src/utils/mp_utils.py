""" 
api for easy multiprocessing
"""
from typing import Callable,List
import multiprocessing as mp
from openai import OpenAI
from tqdm import tqdm
from time import sleep
from rich import print,console
from concurrent.futures import ProcessPoolExecutor

from utils import file_utils

def create_chunk_responces(wrapper:Callable,datas:list):
    """
    This function manages the execution of a wrapper function over a list of data in parallel using multiprocessing.

    Args:
        wrapper (Callable[[list], list]): A function that processes a list of data.
        datas (list): A list containing data to be processed in chunks.

    Returns:
        list: A list of results after processing each data chunk.
    """
    results = []
    with ProcessPoolExecutor(max_workers=len(datas)) as executor:
        results = list(executor.map(wrapper, datas))
    return results



def get_multiple_response(wrapper:Callable,input_datas:list,batch_size = 80,store_fold_path = None,slow=False):
    """
    This function handles batch processing of input data, manages checkpointing, and optionally slows down processing.

    Args:
        wrapper (Callable[[list], list]): The function to apply to each data chunk.
        input_datas (list): A list of dictionaries, each representing an input data point.
        batch_size (int): The number of data points to process in each batch.
        store_fold_path (str, optional): The path to a directory for storing intermediate results for checkpointing.
        slow (bool): If True, adds deliberate delays to processing to manage load or simulate slow server responses.

    Returns:
        list: A list of all results after processing input data.
    """
    outputs = [] # List to hold the output results.

    # Load existing results to avoid reprocessing.
    if store_fold_path:
        temp_jp = file_utils.JsonlProcessor(store_fold_path)
        # reload_data
        outputs = temp_jp.load_lines()
        reload_set = set()
        
        for output in outputs:
            reload_set.add(output["id"])
        #print(reload_set)
        reload_datas = []
        for input_data in input_datas:
            if input_data["id"] not in reload_set:
                reload_datas.append(input_data)
        input_datas = reload_datas
        print(f"断点续连: 这个阶段还要有{len(input_datas)}")
        #temp_jp.dump_restart()
        
    # Calculate the number of chunks needed.
    chunk_size = len(input_datas)//batch_size if len(input_datas) % batch_size==0 else len(input_datas)//batch_size + 1
    
    for i in tqdm(range(chunk_size)):
        if (i+1) * batch_size <= len(input_datas):
            chunk_datas = input_datas[i * batch_size: (i + 1) * batch_size]
        else:
            chunk_datas = input_datas[i*batch_size:len(input_datas)]
        #chunk_results = create_chunk_responces(,chunk_datas)
        # Attempt to process data chunk with retries and error handling.
        for i in range(100):
            try:
                chunk_results = create_chunk_responces(wrapper,chunk_datas)
                if slow:
                    sleep(15)
                break
            except Exception as e:
                console.Console().log(f"[red]can't reach the server!,caused by {e}")
                if slow:
                    sleep(120+i*120)

        new_chunk_results = []
        # !我们认为每个输出返回的是一个list   
        # Flatten the list of results since each returned value is expected to be a list.
        new_chunk_results = [item for sublist in chunk_results for item in sublist]
        outputs.extend(new_chunk_results)
        if store_fold_path:
            temp_jp.dump_lines(new_chunk_results)
            
    # Final cleanup if storing results.
    if store_fold_path:
        temp_jp.close()
    return outputs