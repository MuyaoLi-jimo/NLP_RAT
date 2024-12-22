# <center> 结合COT和RAG的LLM 推理能力优化 </center>

- 团队成员：李沐遥、朱冉


## 项目背景：
在之前的作业中，我们已经尝试通过Retrieval-Augmented Generation（RAG）系统来整合外部知识，以增强Large Language Model（LLM）的世界知识。同时我们还认识到使用Chain of Thought（COT）技术同样有助于提高模型的推理能力。基于此，我们计划进一步探结合COT和RAG，以提高模型在长程上的处理能力。

## 项目目标：
本项目的主要目标是探索将RAG与COT技术相结合的新方法。我们设想先用COT对instruction进行分解，然后对每一步使用RAG进行修改[^1]，期望通过这种方式能够更有效地利用外部知识，从而提升模型在解决长程问题上的准确性。

## 计划方法：
1. 实验框架准备：首先准备好用于实验的数据集（GSM8K,GSM-Hard,Humaneval,MBPP）和baseline（RAG,COT）
2. 链式推理集成：利用langchain API，在RAG系统中集成COT技术，设计并实现prompt模板，用于引导模型进行更加深入的问题分析。
3. 流水线优化：通过改进prompt模版，增加Few Shots Learning，Finetune，改进RAG数据源等技术提高流水线性能
4. 实验与评估：利用准确率，pass@k等指标来对本次实验使用方法进行评估。

## 预期成果：
我们预期该项目能够显著提升语言模型在处理长程任务的性能。

---
[^1] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, Yitao Liang., “RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation” (arXiv, March 8, 2024), https://doi.org/10.48550/arXiv.2403.05313.
