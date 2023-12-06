import string
import re

SUM_FIRST = \
"""
<Summary>
{summary}
</Summary>

<Article>
{article}
</Article>
"""

DOC_FIRST = \
"""
<Article>
{article}
</Article>

<Summary>
{summary}
</Summary>
"""

CONSISTENT_STRING = \
"""1. The summary statements are all supported by the article.

Therefore, the answer is yes, the summary is consistent with the article.
"""

INCONSISTENT_STRING = "Therefore, the answer is no, the summary is not consistent with the article."

SELFINST_TEMPLATE = \
"""<s>[INST] Note that consistency means all information in the summary is supported by the article. 
It's known that the following summary is not consistent with the article. 
Find out why.
{input}
Explain your reasoning step by step:[/INST]
"""

SELFINST_TEMPLATE_EXTRA = \
"""<s>[INST] Note that consistency means all information in the summary is supported by the article. 
It's known that the following summary is not consistent with the article. 
Find out why.
{input}
{extra}
Explain your reasoning step by step:[/INST]
"""

XSUM_EXTRA_TEMPLATE = \
"""For your information, it's reported that the following spans in the summary have consistency issues:
{xsum_annotation}
"""


def inst_parse(input:str, filter=False):
    generation_part = input.partition("[/INST]")[2].partition("###Corrected:")[0]
    reasoning_part = generation_part.partition("\n\n")[2].strip()
    if not filter: return reasoning_part
    reasoning = reasoning_part.split("\n")
    reasoning = [item for item in reasoning if len(item.strip()) > 0 and (not "summary does not" in item)]
    results = []
    for idx, item in enumerate(reasoning, start=1):
        modified_string = re.sub(r'^\d+\.\s*', '', item)
        modified_string = f"{idx}. {modified_string}"
        results.append(modified_string)
    return "\n".join(results)

def annot_parse(input:str):
    corretion_part = input.partition("###Corrected:")[2].partition("###")[0].strip()
    return corretion_part

ZEROSHOT_TEMPLATE = \
"""<s>[INST] Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article. 
{input}
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
{input}
Explain your reasoning step by step first, and then answer (yes or no) the question in the end:[/INST]
"""

def cot_parse(input:str, default = 1, debug=False):
    generation_part = input.partition("[/INST]")[2].lower()
    answer_part = generation_part.strip().split("\n")
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
            if debug:
                import sys
                print("###", answer, file=sys.stderr)
    return default