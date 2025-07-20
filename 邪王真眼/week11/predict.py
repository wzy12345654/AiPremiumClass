import torch

from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer


if __name__ == "__main__":
    model_path = "./邪王真眼/week11/result/checkpoint-2000"
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        aggregation_strategy="simple"
    )

    input_text = "双方确定了今后发展中美关系的指导方针。"

    results = ner_pipeline(input_text)

    output = [
        {
            "entity": entity["entity_group"],
            "content": entity["word"]
        }
        for entity in results
        if entity["entity_group"] != "O"
    ]

    print("输入:", input_text)
    print("输出:", output)