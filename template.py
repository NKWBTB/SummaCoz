import string

CONSISTENT_STRING = \
"""1. The summary statements are all supported by the article.

Therefore, the answer is yes, the summary is consistent with the article.
"""

INCONSISTENT_STRING = "Therefore, the answer is no, the summary is not consistent with the article."

SELFINST_TEMPLATE = \
"""<s>[INST] Note that consistency means all information in the summary is supported by the article. 
It's known that the following summary is not consistent with the article. 
Find out why.

Summary: 
{summary} 

Article: 
{article} 

Explain your reasoning step by step:[/INST]
"""

def inst_parse(input:str):
    generation_part = input.partition("[/INST]")[2].partition("###Corrected:")[0]
    reasoning_part = generation_part.strip().partition(":")[2].strip()
    return reasoning_part

ZEROSHOT_TEMPLATE = \
"""<s>[INST] Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article. 

Article: 
{article} 

Summary: 
{summary} 

Answer (yes or no):[/INST]
"""

def swap_punctuation(input:str):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return input.translate(translator)

def zeroshot_parse(input:str):
    generation_part = input.partition("[/INST]")[2].lower()
    words = swap_punctuation(generation_part).split()
    assert ("consistent" in words) or ("not" in words) or ("no" in words) or ("yes" in words)
    return 0 if ("no" in words) or ("not" in words) else 1
    
COT_TEMPLATE =\
"""<s>[INST] Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article. 

Article: 
{article} 

Summary: 
{summary} 

Explain your reasoning step by step first, and then answer (yes or no) the question in the end:[/INST]
"""

def cot_parse(input:str):
    generation_part = input.partition("[/INST]")[2].lower()
    answer_part = generation_part.strip().split("\n\n")
    for answer in reversed(answer_part):
        words = swap_punctuation(answer).split()
        try:
            assert ("consistent" in words) or \
                ("not" in words) or \
                ("no" in words) or \
                ("yes" in words) or \
                ("accurately" in words) or \
                ("accurate" in words)
            return 0 if ("no" in words) or ("not" in words) else 1
        except:
            import sys
            print("###", answer, file=sys.stderr)
    return 1