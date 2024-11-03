####################
#	1/10
####################

%pip install ../input/hf-peft/peft-0.9.0-py3-none-any.whl
%pip install ../input/bitsandbytes/bitsandbytes-0.42.0-py3-none-any.whl
# %pip install ../input/sentence-transformers/sentence_transformers-2.5.1-py3-none-any.whl
%pip install ../input/transformers-4-39-2/transformers-4.39.2-py3-none-any.whl



####################
#	2/10
####################

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm.auto import tqdm
import torch
import pandas as pd
from tqdm import tqdm
import json



####################
#	3/10
####################

test = pd.read_csv("../input/llm-prompt-recovery/test.csv")
!cp ../input/llm-prompt-recovery/test.csv .



####################
#	4/10
# date_reading: 
# thought: 
# words: 
# reference: 

####################

%%writefile run.py

# !cp ../input/recovery-scripts/run.py .
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm
import json
from peft import PeftModel, PeftConfig
import argparse
import numpy as np

# Create the argument parser
parser = argparse.ArgumentParser(description="")

parser.add_argument("--model_path", type=str, help="")
parser.add_argument("--peft_path", type=str, help="", default="")
parser.add_argument("--model_type", type=str, help="")
parser.add_argument("--prime", type=str, help="", default="")
parser.add_argument("--magic", type=str, help="", default="")
parser.add_argument("--output", type=str, help="")
parser.add_argument("--max_len", type=int, help="")
parser.add_argument("--min_output_len", type=int, help="", default=2)
parser.add_argument("--max_output_len", type=int, help="", default=100)
parser.add_argument('--quantize', action='store_true')
parser.add_argument('--do_sample', action='store_true')
parser.add_argument('--test_path', type=str)

args = parser.parse_args()

test = pd.read_csv(args.test_path)
magic = "Transform the following text in a more vivid and descriptive way, while maintaining the original meaning and tone."
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
lucrarea = args.magic

def _predict_gemma(row: pd.Series, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, bad_words_ids: list) -> str:
    if row.original_text == row.rewritten_text:
        return "Correct grammatical errors in this text."
    ot = " ".join(str(row.original_text).split(" ")[:args.max_len])
    rt = " ".join(str(row.rewritten_text).split(" ")[:args.max_len])
    prompt = f"Find the orginal prompt that transformed original text to new text.\n\nOriginal text: {ot}\n====\nNew text: {rt}"
    conversation = [{"role": "user", "content": prompt }]
    prime = args.prime
    # apply_chat_template returns string when tokenize is False.
    # [Chat Templates](https://huggingface.co/docs/transformers/main/ja/chat_templating)
    # [Utilities for Tokenizers](https://huggingface.co/docs/transformers/main/ja/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)
    prompt = tokenizer.apply_chat_template(
        conversation = conversation, 
        tokenize=False
    ) + f"<start_of_turn>model\n{prime}"
    # [Tokenizer — transformers 2.11.0 documentation](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode)
    # about truncation: [transformersのTokenizerで固定長化する - Money Forward Developers Blog](https://moneyforward-dev.jp/entry/2021/10/05/transformers-tokenizer/)
    # Encoded
    # 入力テキストをトークン化し、テンソル形式で取得
    input_ids = tokenizer.encode(
        prompt,  # 入力となるプロンプト文字列
        add_special_tokens=False,  # 特殊トークンを追加しない
        truncation=True,  # 最大長を超える部分を切り捨てる
        max_length=1536,  # 最大トークン数を1536に設定
        padding=False,  # パディングを行わない
        return_tensors="pt"  # PyTorchのテンソル形式で返す
    )
    # [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
    # Encoded
    x = model.generate(
        input_ids=input_ids.to(model.device),  # モデルのデバイス（CPU/GPU）に合わせて入力を転送
        eos_token_id=tokenizer.eos_token_id,  # 生成終了を示すトークンID
        pad_token_id=tokenizer.eos_token_id,  # パディングに使用するトークンID
        max_new_tokens=128,  # 新たに生成する最大トークン数を128に設定
        do_sample=args.do_sample,  # サンプリングを行うかどうか（True/False）
        early_stopping=True,  # 早期終了を有効にする
        num_beams=1,  # ビームサーチのビーム数を1に設定（グリーディーサーチ）
        bad_words_ids=bad_words_ids  # 生成時に避けるべきトークンIDのリスト
    )
    try:
        # 生成されたトークンIDをデコードして文字列に変換
        x = tokenizer.decode(
            x[0] # Tensorを配列にする（num_return_sequencesを設定している場合ははじめのものを取得する意味になる）
        ).split("<start_of_turn>model")[1].split("<end_of_turn>")[0].replace("<end_of_turn>\n<eos>","").replace("<end_of_turn>","").replace("<start_of_turn>","").replace("<eos>","").replace("<bos>","").strip().replace('"','').strip()
        x = x.replace("Can you make this","Make this").replace("?",".").replace("Revise","Rewrite")
        x = x.split(":",1)[-1].strip()
        if "useruser" in x:
            x = x.replace("user","")
        # https://docs.python.org/3/library/stdtypes.html#str.isalnum
        if x[-1].isalnum():
            x += "."
        else:
            x = x[:-1]+"."
        x+= lucrarea
        if len(x.split()) < args.max_output_len and len(x.split()) > args.min_output_len and ("\n" not in x):
            print(x)
            return x
        else:
            return magic
    except Exception as e:
        print(e)
        return magic

def predict_gemma(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, test: pd.DataFrame, bad_words_ids=None) -> list[str]:
    if bad_words_ids is not None and len(bad_words_ids) == 0:
        bad_words_ids = None
    predictions = []
    with torch.no_grad():
        for _, row in tqdm(test.iterrows(), total=len(test)):
            prediction = _predict_gemma(row, model, tokenizer, bad_words_ids)
            predictions.append(prediction)
    return predictions

def _predict_mistral(row: pd.Series, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prime: str) -> str:
    ot = " ".join(str(row.original_text).split(" ")[:args.max_len])
    rt = " ".join(str(row.rewritten_text).split(" ")[:args.max_len])
    prompt = f'''
Please find the prompt that was given to you to transform **original_text** to **new_text**. One clue is the prompt itself was short and concise.
Answer in thist format: "It's likely that the prompt that transformed original_text to new_text was: <the prompt>" and don't add anything else.

**original_text**:
{ot}

**new_text**:
{rt}
'''
    conversation = [{"role": "user", "content": prompt }]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)+prime
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=1536,padding=False,return_tensors="pt")
    x = model.generate(input_ids=input_ids.to(model.device), eos_token_id=[13, tokenizer.eos_token_id], pad_token_id=tokenizer.eos_token_id, max_new_tokens=32, do_sample=args.do_sample, early_stopping=True, num_beams=1)
    try:
        x = tokenizer.decode(x[0]).split("[/INST]")[-1].replace("</s>","").strip().split("\n",1)[0]
        x = x.replace("Can you make this","Make this").replace("?",".")
        # print(x.split(":",1)[0])
        x = x.split(":",1)[-1].strip()
        if x[-1].isalnum():
            x += "."
        else:
            x = x[:-1]+"."
        x += lucrarea
        predict = None
        if len(x.split()) < 50 and len(x.split()) > 2 and ("\n" not in x):
            predict = x
        else:
            predict = magic
        print(predict)
        return predict
    except Exception as e:
        print(e)
        return magic

def predict_mistral(model, tokenizer, test,prime="") -> list[str]:
    predictions = []
    with torch.no_grad():
        for _, row in tqdm(test.iterrows(), total=len(test)):
            prediction = _predict_mistral(row, model, tokenizer, prime)
            predictions.append(prediction)
    return predictions

model_name = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
banned_ids = None
    
if args.quantize:
    print("Use 4bit quantization")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=quantization_config,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    if args.peft_path != "":
        print("Use peft")
        # [Models](https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel)
        # 指定されたパスからPEFTの設定を読み込み、ベースモデルに適用
        model = PeftModel.from_pretrained(model,
                                    args.peft_path,
                                    quantization_config=quantization_config,
                                    torch_dtype=torch.bfloat16,
                                    device_map="auto")
        # [Configuration](https://huggingface.co/docs/peft/main/en/package_reference/config#peft.PeftConfig)
        # [gemma 7b orca 68500](https://www.kaggle.com/datasets/suicaokhoailang/gemma-7b-orca-68500/data)
        # [LoRA](https://huggingface.co/docs/peft/package_reference/lora)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    if args.peft_path != "":
        print("Use peft")
        model = PeftModel.from_pretrained(model,
                                args.peft_path,
                                torch_dtype=torch.bfloat16,
                                device_map="auto")
        
# model = model.merge_and_unload()
model.eval()
# print(model)
if args.model_type == "gemma":
    preds = predict_gemma(model, tokenizer, test, bad_words_ids=banned_ids)
elif args.model_type == "mistral":
    preds = predict_mistral(model, tokenizer, test, prime=args.prime)

json.dump(preds, open(args.output,"wt"))



####################
#	5/10
# date_reading: 
# thought: 
# words: 
# reference: 

####################

!python run.py --model_path /kaggle/input/gemma/transformers/7b-it/3/ --peft_path "../input/gemma-7b-orca-68500/" --model_type "gemma" --output "pred2.json" --max_len 512 --test_path ./test.csv --quantize --prime "General prompt: Improve this text using the writing style"
preds = json.load(open("pred2.json"))
# preds = ["Please improve this text using the writing style with maintaining the original meaning but altering the tone.",]*len(test)
def remove_pp(x):
    for w in x.split()[1:]:
        if w.istitle():
            return "Please improve this text using the writing style."
    return x
preds = [remove_pp(x)[:-1]+" with maintaining the original meaning but altering the tone." for x in preds]
json.dump(preds, open("pred2.json","wt"))



####################
#	6/10
# date_reading: 
# thought: 
# words: 
# reference: 

####################

!python run.py --model_path /kaggle/input/mistral/pytorch/7b-v0.1-hf/1 --peft_path "../input/mistral-og-600" --model_type "mistral" --output "pred0.json" --max_len 512 --test_path ./test.csv --quantize --prime "It's likely that the prompt that transformed original_text to new_text was: Rewrite" --magic ""



####################
#	7/10
# date_reading: 
# thought: 
# words: 
# reference: 

####################

!python run.py --model_path ../input/mistral-7b-it-v02/ --peft_path "../input/mistral-gemmaonly" --model_type "mistral" --output "pred3.json" --max_len 512 --test_path ./test.csv --quantize --prime "It's likely that the prompt that transformed original_text to new_text was: Make this text" --magic ""



####################
#	8/10
# date_reading: 
# thought: 
# words: 
# reference: 

####################

!python run.py --model_path  /kaggle/input/gemma/transformers/7b-it/3 --peft_path "../input/gemma-7b-orca-external/" --model_type "gemma" --output "pred1.json" --max_len 512 --test_path ./test.csv --quantize --prime "General prompt: Alter" --magic ""



####################
#	9/10
# date_reading: 
# thought: 
# words: 
# reference: 

####################

fns = ["pred0.json","pred1.json", "pred2.json", "pred3.json"]
preds = [json.load(open(x)) for x in fns]
preds = [' '.join(list(x)) for x in zip(*preds)]
print(preds[:5])



####################
#	10/10
# date_reading: 
# thought: 
# words: 
# reference: 

####################

magic = " 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"
# magic = ""
predictions = [x+magic for x in preds]

sub = pd.read_csv("../input/llm-prompt-recovery/sample_submission.csv")
sub['rewrite_prompt'] = predictions
sub.to_csv('submission.csv',index=False)
