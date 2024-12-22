import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from rich import print
import time
from pathlib import Path
from src.utils import file_utils

def crawl(url_group_name:str,):
    # 获取要爬取的网页
    url_group_index_path = Path("data/crawl_docs/url_group_index.json")
    url_group = file_utils.load_json_file(url_group_index_path).get(url_group_name)
    if url_group is None:
        raise AssertionError(F"{url_group} is empty")
    
    datas = []
    
    # 尝试两次爬取
    for _ in range(2):
        new_datas,url_group = get_contents(url_group)
        datas.update(new_datas)
    print(f"unable to visit: {url_group}")
    
    # 保存
    url_group_fold = Path(f"data/crawl_docs/{url_group_name}")
    url_raw_data = url_group_fold/"raw"
    url_group_path.mkdir(parents=True,exist_ok=True)
    url_raw_data.mkdir(parents=True,exist_ok=True)
    url_group_path = url_group_fold/"_index.json"
    
    
    url_index = [data["title"] for data in datas.values()]
    file_utils.dump_json_file(url_index,url_group_path)
    for data in tqdm(datas.values()):
        content_path = url_raw_data / f"{data['title']}.html"
        content = datas["html_content"]
        with open(content_path, "w", encoding="utf-8") as file:
            file.write(content)

def get_contents(urls):
    """
    collects HTML content and titles from a list of URLs.

    :param urls: List of URLs to be crawled
    :return: a dictionary of data collected from URLs 
             a list of unvisited URLs
    """
    visited_urls = []   #已经爬取
    unvisited_urls = [] #爬取失败
    datas = {} #采集到的数据
    for url in tqdm(urls,desc="collect html"):
        # 如果之前已经存过，则跳过
        if url in visited_urls:
            continue
        try:
            print(f'current url: {url}')
            success, title, html_content = get_content(url)
            if success:
                data = {
                    'url': url,
                    'title': title,
                    'html_content': html_content    
                }
                datas[url] = data
                visited_urls.append(url)
                time.sleep(8)
            else:
                unvisited_urls.append(url)
        except Exception as e:
            print("error: ", e)
            unvisited_urls.append(url)
            time.sleep(2)
    return datas,unvisited_urls

#获取标题和内容
def get_content(url):
    """
    Fetches the content from a given URL and extracts the title and HTML content if successful.

    :param url: URL to fetch content from
    :return: (success flag, title of the page,HTML content,)
    """
    # Fetch the content from the URL
    response = requests.get(url)

    # Check if the request was successful
    print(response.status_code)
    if response.status_code == 200:
        # Extract the HTML content
        html_content = response.text

        # Initialize the BeautifulSoup object
        soup = BeautifulSoup(html_content, 'html.parser')
        
        title = soup.title.text #把该页标题打出

        return True, title, html_content
    else:
        print("Failed to fetch the content.")
        return False, None, None
    
if __name__ == "__main__":
    crawl("chemistry")
    