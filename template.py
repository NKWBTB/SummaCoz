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

ZEROSHOT_TEMPLATE = \
"""<s>[INST] Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article. 

Article: 
{article} 

Summary: 
{summary} 

Answer (yes or no):[/INST]
"""

def zeroshot_parse(input:str):
    generation_part = input.partition("[/INST]")[2].lower()
    return 0 if ("no" in generation_part) or ("not" in generation_part) else 1
    
COT_TEMPLATE =\
"""<s>[INST] Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article. 

Article: 
{article} 

Summary: 
{summary} 

Explain your reasoning step by step first, and then answer (yes or no) the question in the end:[/INST]
"""