from collections import namedtuple
from torch.utils.data import Dataset
from datasets import load_from_disk

ChatMessage = namedtuple("ChatMessage", ["role", "content", "ipython", "eot" ,"masked"])

class ProcessedDataset(Dataset):
    def __init__(self, tokenizer=None, packed=False):
        self.dataset = load_from_disk("/content/processed_dataset")
        self.tokenizer = tokenizer
        self.packed = packed

    def __getitem__(self, idx):
        row = self.dataset[idx]

        # For each message, the .content is now a list of items, each a dict with "type" and "content"
        user_msg = ChatMessage(
            role="user",
            content=[
                {"type": "text", "content": row["instruction"]},
            ],
            ipython=False,  # or True if needed
            eot=False,
            masked=False
        )
        assistant_msg = ChatMessage(
            role="assistant",
            content=[
                {"type": "text", "content": row["response"]},
            ],
            ipython=False,  # or True if needed
            eot=True,
            masked=False
        )

        sample = {"messages": [user_msg, assistant_msg]}

        # Now call the tokenizer (which expects a typed body)
        tokenized = self.tokenizer(sample)
        
        # print("DEBUG: tokenizer output ->", tokenized)
        # raise RuntimeError("Stop here to inspect output")
        return {"tokens": tokenized["tokens"],       
                "labels": tokenized["tokens"]
                }

    def __len__(self):
        return len(self.dataset)
