from pprint import pprint
from datasets import load_dataset
# 日本語のLLM評価用データセットをロード (エラーが出ていたが、Python: Select Interpreterでvenvを選択したら解決した)
# ダウングレードしないとエラーになる

dataset = load_dataset("llm-book/llm-jp-eval", "jamp")
pprint(dataset["train"][0])
print("ラベルの種類:", set(dataset["train"]["output"]))
