from pprint import pprint
from transformers import pipeline

text2text_pipeline = pipeline(
    model="llm-book/t5-base-long-livedoor-news-corpus"
)
article = "ついに始まった３連休。テレビを見ながら過ごしている人も多いのではないだろうか？　今夜オススメなのは何と言っても、NHKスペシャル「世界を変えた男 スティーブ・ジョブズ」だ。実は知らない人も多いジョブズ氏の養子に出された生い立ちや、アップル社から一時追放されるなどの経験。そして、彼が追い求めた理想の未来とはなんだったのか、ファンならずとも気になる内容になっている。 今年、亡くなったジョブズ氏の伝記は日本でもベストセラーになっている。今後もアップル製品だけでなく、世界でのジョブズ氏の影響は大きいだろうと想像される。ジョブズ氏のことをあまり知らないという人もこの機会にぜひチェックしてみよう。 世界を変えた男　スティーブ・ジョブズ（NHKスペシャル）"
# articleの要約を生成
print(text2text_pipeline(article)[0]["generated_text"])
