import openai
import utils
import template
import os
import time
import keys
from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

def consistency_check(item,client,model_name, role="system"):
    doc_sum = template.COT_TEMPLATE.format(article = item["document"],summary = item["claim"])
    chat_completion = completion_with_backoff(
        client=client,
        model=model_name,
        messages=[{"role": role, "content": doc_sum}],
        temperature=0,
        max_tokens = keys.MAX_NEW_TOKEN
    )
    step_reasoning = chat_completion.choices[0].message.content
    item[model_name] = step_reasoning
    return item

def main(api_key, role, model_name="gpt-3.5-turbo-0301", base_url=None):
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    benchmark = utils.load_jsonl("benchmark_v_0_6.jsonl")
    for index,item in enumerate(tqdm(benchmark)):
        file_name = os.path.join(model_name, f"{index}.jsonl")
        if os.path.exists(file_name): continue
        result = consistency_check(item,client,model_name, role=role)
        utils.dump2jsonl(lines = [result], output_path = file_name)
        time.sleep(60.0/keys.RPM[model_name])

if __name__ == "__main__":
    # main()
    main(keys.anyscale_key,
         role="user",
         base_url="https://api.endpoints.anyscale.com/v1",
         model_name="meta-llama/Llama-2-7b-chat-hf")
    
    

# benchmark_v_0_5 = utils.load_jsonl("benchmark_v_0_6.jsonl")
# doc = benchmark_v_0_5[0]["document"]

# sum = benchmark_v_0_5[0]["claim"]

# tem = template.COT_TEMPLATE.format(article=doc, summary=sum)


# system_content = tem
# client = openai.OpenAI(api_key="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")
# chat_completion = client.chat.completions.create(
#     model="gpt-3.5-turbo-1106",
#     messages=[{"role": "system", "content": system_content}],
#     temperature=0,
#     max_tokens = 500
# )
# product_names = chat_completion.choices[0].message.content
# print("Results from the OpenAI endpoint:\n", product_names)