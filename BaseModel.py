from typing import Optional
import time

from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)
from llama_models.llama3.reference_impl.generation import Llama

class BaseModel:
    def __init__(self,
        ckpt_dir: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 2048,
        max_batch_size: int = 4,
        max_gen_len: Optional[int] = None,
        model_parallel_size: Optional[int] = None,
        keep_history = False
    ):
        
        self.ckpt_dir = ckpt_dir
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_gen_len = max_gen_len
        self.model_parallel_size = model_parallel_size
        self.keep_history = keep_history

        if self.keep_history:
            self.dialog = []

        tokenizer_path = str("../llama-models/models/llama3/api/tokenizer.model")
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )

    def forward(self, message: str):
        dialog = self.dialog if self.keep_history else []
        user_message = UserMessage(content=message)
        dialog.append(user_message)
        result = self.generator.chat_completion(
            dialog,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        out_message = result.generation
        # print(f"> {out_message.role.capitalize()}: {out_message.content}")
        if self.keep_history:
            dialog.append(out_message)
        return out_message


# run with 
# CHECKPOINT_DIR=/checkpoints/saner/.llama/checkpoints/Llama3.2-3B-Instruct
# CUDA_VISIBLE_DEVICES=7 torchrun BaseModel.py
if __name__ == "__main__":
    prompt = """
Summarize this dialog:
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
---
Summary:
"""
    # prompt = "Can I register a car to multiple owners in california"
    base_agent = BaseModel("/checkpoints/saner/.llama/checkpoints/Llama3.2-1B/")
    print(base_agent.forward(prompt).content)