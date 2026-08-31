# P1-EVAL-MIN 意图评测报告

- 评测使用真实 DeepSeek 意图分类器。
- 评测同时检查分类标签和生产置信度门控结果。
- 这是 20 条人工标注固定题集的单次在线评测结果。
- 结果：20/20 通过，accuracy 100.0%。
- label_mismatch：0。
- gate_mismatch：0。
- errors：0。
- 题集分布：allow 8 / unrelated 8 / uncertain 4。
- 该结果不代表真实生产分布或总体准确率。
- 本评测不覆盖检索、Reranker、回答生成和引用语义正确性。
- API Key 仅由评测进程从环境变量读取，未写入题集、报告或控制台输出。
