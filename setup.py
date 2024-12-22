from setuptools import setup, find_packages

setup(
    name='final_proj',  #名字
    version='0.0.1',           #版本
    author='jimoli',        #作者 
    description='using method cot',  #描述
    packages=find_packages(
        exclude=["logs*", "doc*", "data*","lib*"]
        ),  #你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包 
)
