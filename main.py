from transformers import DistilBertModel

model = DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
print(model)