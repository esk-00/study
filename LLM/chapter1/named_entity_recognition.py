from pprint import pprint
from transformers import pipeline

ner_pipline = pipeline(
  model = "llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
  aggregation_strategy="simple"
)
text = "大谷正平は岩手県水沢市出身のプロ野球選手"
# text中の固有表現を抽出
pprint(ner_pipline(text))
