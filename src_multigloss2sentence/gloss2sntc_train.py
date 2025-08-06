# https://www.vennify.ai/fine-tune-grammar-correction/
# https://happytransformer.com/save-load-model/
# https://www.youtube.com/watch?v=sQY-4lP3XoE
# happytransformer==3.0.0 tf-keras


# import csv
from happytransformer import HappyTextToText, TTTrainArgs
# from happytransformer import HappyTextToText
from happytransformer import TTSettings
# from datasets import load_dataset


# train_dataset= load_dataset("jfleg", split='validation[:]')
# eval_dataset= load_dataset("jfleg", split='test[:]')
# def remove_excess_spaces(text):
#     replacements= [
#         (" .", "."), 
#         (" ,", ","),
#         (" '", "'"),
#         (" ?", "?"),
#         (" !", "!"),
#         (" :", "!"),
#         (" ;", "!"),
#         (" n't", "n't"),
#         (" v", "n't"),
#         ("2 0 0 6", "2006"),
#         ("5 5", "55"),
#         ("4 0 0", "400"),
#         ("1 7-5 0", "1750"),
#         ("2 0 %", "20%"),
#         ("5 0", "50"),
#         ("1 2", "12"),
#         ("1 0", "10"),
#         ('" ballast water', '"ballast water')
#     ]
#     for rep in replacements:
#         text= text.replace(rep[0], rep[1])
#     return text
# def generate_csv(csv_path, dataset):
#     with open(csv_path, 'w', newline='') as csvfile:
#         writter= csv.writer(csvfile)
#         writter.writerow(["input", "target"])
#         for case in dataset:
#             # Adding the task's prefix to input 
#             input_text= "grammar: " + case["sentence"]
#             for correction in case["corrections"]:
#                 # a few of the cases contain blank strings. 
#                 if input_text and correction:
#                     input_text= remove_excess_spaces(input_text)
#                     correction= remove_excess_spaces(correction)
#                     writter.writerow([input_text, correction])
# generate_csv("train.csv", train_dataset)
# generate_csv("eval.csv", eval_dataset)
# t5finetuned_gram= HappyTextToText("T5", "t5-base")
t5finetuned_gram= HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args= TTTrainArgs(batch_size=1)
print(f"before {t5finetuned_gram.eval("./eval.csv").loss}")
t5finetuned_gram.train("./train.csv", args=args)
print(f"after {t5finetuned_gram.eval("./eval.csv").loss}")


# t5finetuned_gram= HappyTextToText("T5", "vennify/t5-base-grammar-correction")
# t5finetuned_gram= HappyTextToText("T5", "model_sent/")
beam_settings=  TTSettings(num_beams=5, min_length=1, max_length=20)
print(t5finetuned_gram.generate_text(
    "grammar: I boughts ten apple.",
    args=beam_settings).text
)
print(t5finetuned_gram.generate_text(
    "grammar: This sentences, has bads grammar and spelling!",
    args=beam_settings).text
)


t5finetuned_gram.save("model_sent/")

