from pprint import pprint
from datasets import load_dataset
# 日本語のLLM評価用データセットをロード (エラーが出ていたが、Python: Select Interpreterでvenvを選択したら解決した)
# ダウングレードしないとエラーになる

# ここで指定したファイルで'cp932"のエラーが出る場合はそのフォルダまでいき、.open(encoding='utf-8')にする
# jnliの場合、呼び出すファイルを適切なものに変更する必要がある。github(https://github.com/yahoojapan/JGLUE/tree/main/datasets)を参照し、rawボタンを押下した後のURL
dataset = load_dataset("llm-book/llm-jp-eval", "jnli")
pprint(dataset["train"][0])
print("ラベルの種類:", set(dataset["train"]["output"]))
