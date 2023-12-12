DOC_FIRST = \
"""
<Article>
{article}
</Article>

<Summary>
{summary}
</Summary>
"""

POSTHOC_TEMPLATE = \
f"""Note that consistency means all information in the summary is supported by the article. 
It's known that the following summary is not consistent with the article. 
Find out why.
{DOC_FIRST}
Explain your reasoning step by step:[/INST]
"""

COT_TEMPLATE = \
f"""Decide if the following summary is consistent with the corresponding article. 
Note that consistency means all information in the summary is supported by the article. 
{DOC_FIRST}
Explain your reasoning step by step first, and then answer (yes or no) the question in the end:
"""

GRADER_TEMPLATE = \
"""You are an assessor to give judgement on a reasoning problem.
Here is the text to be assessed:

<text>
{input_reasoning}
</text>

Does the above text mention or contain the following reference reasoning step:

<reference>
{reference}
</reference>

Answer (yes or no):
"""