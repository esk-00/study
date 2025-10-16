from transformers import pipeline

nli_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-jnli")
text = "二人の男性がジェット機を見ています"
entailment_text = "ジェット機を見ている人がいます"

#textとentailment_textの論理関係を予測
print(nli_pipeline({"text": text, "text_pair": entailment_text}))

contradiction_text = "二人の男性が飛んでいます"
#textとcontradiction_textの論理関係を予測
print(nli_pipeline({"text": text, "text_pair": contradiction_text}))

neutral_text = "二人の男性が、白い飛行機を眺めています"
#textとneutral_textの論理関係を予測
print(nli_pipeline({"text": text, "text_pair": neutral_text}))
