import string
import re

CLAIM_PROMPT = \
"""
<Article>
{article}
</Article>

<Claim>
{summary}
</Claim>
"""

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

CONSISTENT_STRING = "Therefore, the answer is yes, the summary is consistent with the article."

CONSISTENT_TEMPLATE = \
"""1. The summary statements are all supported by the article.
{consistent_string}
""".format(consistent_string=CONSISTENT_STRING)

INCONSISTENT_STRING = "Therefore, the answer is no, the summary is not consistent with the article."

SELFINST_TEMPLATE = \
"""<s>[INST] Note that consistency means all information in the summary is supported by the article. 
It's known that the following summary is not consistent with the article. 
Find out why.
{input}
Explain your reasoning step by step:[/INST]
"""

SELFINST_TEMPLATE_POSITIVE = \
"""<s>[INST] You are given an article and a claim.
Please read the <Claim> and find the sentence(s) in the <Article> that express the <Claim>.
{input}
Your reponse:[/INST]

1. The article states that"""

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

TYPE_EXTRA_TEMPLATE = \
"""For your information, it's reported that the sumary has following issues:
{type_annotation}
"""

POLYTOPE_TYPES = {
    "Positive_Negative_Aspect": "Aspect Error: The summary content represents the positive aspect whereas the article is negative and vice versa. For example, if a summarized text tells the user to push a button when the article tell the user not to push it, there is an Aspect Error.",
    "Inaccuracy_internal": "Intrinsic Error: Terms or concepts from the source are misrepresented and thus unfaithful. For example, \"Pittsburgh Union Station is 10 kilometers from Exhibition Center and 3 kilometers from the University of Pittsburgh\" in the article but \"Pittsburgh Union Station is 3 kilometers from Exhibition Center\" in the summary.",
    "Inaccuracy_external": "Extrinsic Error: The summary has content not presented in the source and factually incorrect. For example, it is described as \"Pittsburgh Union Station, also known as Pittsburgh South Station\" in the output but \"Pittsburgh South Station\" is neither mentioned in the source text nor exists in the real world.",
}

FRANK_TYPES = {
    "CorefE": "Coreference Error: A noun/reference with wrong or non-existing antecedent. For example, the summary writes \"The first vaccine for Ebola was approved in 2019. They say a vaccine for COVID-19 isunlikely to be ready this year.\" The pronoun \"They\" here does not have an referent.",
    "LinkE": "Discourse Error: Error in how multiple statements are linked together in the discourse (temporal ordering/causal link). For example, the summary states \"scientists have to show successful human trials, then sequence the DNA of the virus\", while the article states \"scientists had to sequence the DNA of Ebola, then identify possible vaccines, and finally show successful clinical trials\".",
    "GramE": "Grammatical Error: The grammar of the sentence is so wrong that it becomes meaningless. Example: \"The Ebola vaccine accepted have already started.\"",
    "EntE": "Entity Error: The primary arguments (or their attributes) of the predicate are wrong. For example, the summary writes \"the COVID-19 vaccine was approved\", but the article says \"the vaccine for Ebola was approved\".",
    "CircE": "Circumstance Error: The additional information (like location or time) specifying the circumstance around a predicate is wrong. For example, the summary states the first vaccine for Ebola was approved by the FDA in 2014, but the article states 2019.",
    "RelE": "Relation Error: The predicate in the summary statement is inconsistent with the source article. For example, the summary states that the Ebola vaccine was rejected, but the article states that the vaccine was approved.",
    "OutE": "Out of Article Error: The statement contains information not present in the source article. For example, the summary mentions China, while the article does not.",
    "OtherE": "Error: The summary has hallucinated or contains one or more factual errors."
}

def inst_parse(input:str, filter=False):
    generation_part = input.partition("[/INST]")[2].partition("###Corrected:")[0]
    reasoning_part = generation_part.partition("\n\n")[2].strip()
    reasoning_part = reasoning_part.replace("\n\n", "\n")
    reasoning_part = reasoning_part.replace("<Claim>", "claim")
    reasoning_part = reasoning_part.replace("<Article>", "article")
    reasoning_part = reasoning_part.replace("<Summary>", "summary")
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