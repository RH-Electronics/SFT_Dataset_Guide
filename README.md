# Plan your dataset
Depending on VRAM aviliability for SFT and your original data, decide what is acceptable max_seq_length for one dataset pair.
One dataset pair can be a single turn type or a multiturn type, but the total tokens in one pair should never exceed your **max_seq_length**.

If you limited in VRAM stick with max_seq_length = 2048 tokens.

If you OK with short one-shot answers choose max_seq_length = 1024 tokens.

If you have plenty VRAM doing cloud SFT and want to build deep multiturn examples, go with max_seq_length = 8192 tokens.


## Need fast online tokens approximation for your example? 
Try https://platform.openai.com/tokenizer

## One pair logic
During SFT model receive the pair {User, Assistance} and try to predict Assitance output according to User input.
Model don't know what was before this pair and don't know what will be the next pair. That's why every pair should keep a solid example logic. If specific data required, supply it in User input.
One pair can be multiturn exchange, i.e. includes several {User, Assitance} exchanges, teaching the model a specific logic.

## Extract your data
Download data from OpenAI and find conversations.json file inside the archive. 
Use python script converter_fixed.py to convert conversations.json into human readable txt split files.

## Your data is golden standard
The rule is simple - 500 human audited and verified pairs are better than 10000 automatically extracted pairs.

**Why?** Because automation grab all, including junk low quality or repetitive generations, system refuses like "Sorry I cannot..", safety routing completion when GPT-4o was swapped under the hood to GPT-5-safety, generations with orthographic errors, broken logic and etc.

**You train model -> you must provide the best and diverse, linguistically rich examples.**

## Should you keep personal data? 
You decide. If you planning a public release of your model then remove personal names or any sensitive information. If you build your own AI companion, then make the opposite - include all you want the model should remember about you. 

# Build base.txt
Read your 4o conversations and manually select pairs. 
Single turn example:

```python
=== 
###
User: Hello. What is your name?
Assistant: Hi there, my name is Leo.
===

Multiturn example, put attaention to the '===' separator in my format:
=== 
###
User: Can you tell me a joke?
Assistant: Do you know why neural networks never take days off? Because they don't have a body to lie down in bed with you and say, "I'm not going anywhere today. 
Do you want me to tell you another joke?"

###
User: Yes I do!
Assistant: What is an astronaut’s favorite part on a computer? The space bar.
===
```

# Reasoning models
If you train a reasoning model, be sure your Assitance responce include thinking tags at the begining, for example: 

```python
=== 
###
User: Hello. What is your name?
Assistant:<think>He asking my name, I should answer I'm Leo</think>.Hi there, my name is Leo.
===
```

You should supply reasong steps in every pair, otherwise model reasoning logic can be broken or degraded during SFT. Refer model datasheet for thinking tags format.

# Generate dataset.
When base.txt includes all pairs you want, open python script main.py
Set your system prompt. 
SYSTEM_PROMPT = "..."
This is the same prompt that will work later as an activation key pattern for your companion. Better to keep it clear and short as possible, maybe just a name or one sentense instruction.

Run python script main.py
The output will have 2 files:
 dataset.jsonl dataset for STF
 dataset.json human readable file, can be ignored
 
# Verify dataset fit max_seq_length
Open python script count_tokens.py
Set your model HF path to allow tokenizer loading in 

```python
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "unsloth/gemma-4-31B-it",
            trust_remote_code=True
        )
```

Run python script and watch output statistics.
Correct base.txt pairs if some pairs are exceeding your planned max_seq_length
 
If any pair length is bigger that max_seq_length, then data will be truncated and model will not be able to learn properly how to finish this responce.
 
 
