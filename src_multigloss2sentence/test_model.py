from happytransformer import HappyTextToText
from happytransformer import TTSettings


t5finetuned_gram= HappyTextToText("T5", "model_sent/")
beam_settings=  TTSettings(num_beams=5, min_length=1, max_length=20)


incorrect_gram= [
    "where comfort room?",
    "this available?",
    "why life hard?",
    "help, I dying",
    "when time my appointment schedule?",
    "doctor available?",
    "how long wait time",
    "when I get result?",
    "I peanut allergy",
]
for sent in incorrect_gram:
    print(t5finetuned_gram.generate_text(
        sent,
        args=beam_settings).text
    )

