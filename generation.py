import torch
from torch import Tensor
from tqdm import tqdm

from jaxtyping import Int, Float

from llm import softmax, TransformerLM
from tokenizer import Tokenizer

def get_nucleus_sample(probs: Float[Tensor, "vocab_size"], target_p: float) -> Float[Tensor, "n_nucleus"]:
    sorted_values, indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs: Float[Tensor, "vocab_size"] = sorted_values.cumsum(dim=-1)
    cumulative_probs = torch.cat((torch.zeros(1, device=cumulative_probs.device), cumulative_probs[:-1]))
    eligible_indices: Float[Tensor, "n_nucleus"] = indices[(cumulative_probs <= target_p).to(torch.bool)].to(torch.int64)
    result = probs[eligible_indices]
    return result


def generate_text(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_tokens: int, device: str, temperature: float = 1.0, top_p: float = 0.99) -> str:
    response: str = ""
    model.eval()

    input_tokens: Int[Tensor, "seq_len"] = Tensor(tokenizer.encode(prompt)).to(device)
    for _ in tqdm(range(max_tokens), desc=f"generating tokens"):
        logits: Float[Tensor, "seq_len vocab_size"] = model(input_tokens)
        next_token_logits: Float[Tensor, "vocab_size"] = logits[-1,:]
        next_token_logits /= temperature
        probabilities: Float[Tensor, "vocab_size"] = softmax(next_token_logits, dim=-1)
        candidate_probs: Float[Tensor, "n_candidate"] = get_nucleus_sample(probabilities, top_p)
        next_token: int = torch.multinomial(candidate_probs, 1).item() # type: ignore
        response += tokenizer.decode([next_token])

        if tokenizer.is_ending_token(next_token):
            print("terminating after end of text token")
            break

        input_tokens = torch.cat((input_tokens, Tensor([next_token]).to(device)))[-model.params.context_length:]

    return response
