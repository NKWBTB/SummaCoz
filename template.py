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
{Article} 

Summary: 
{Summary} 

Answer (yes or no):[/INST]
"""

COT_TEMPLATE =\
"""<s>[INST] Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article. 

Article: 
{Article} 

Summary: 
{Summary} 

Explain your reasoning step by step then answer (yes or no) the question:[/INST]
"""