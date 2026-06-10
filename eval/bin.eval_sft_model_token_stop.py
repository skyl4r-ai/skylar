import torch
from tokenizers import Tokenizer, decoders
from models.decoder import NanoTransformer
from utils.chatML import encode_chatml

model = NanoTransformer.from_pretrained('checkpoints/skylar-100M-Chat-v4/best').to('cuda').eval()
tok = Tokenizer.from_file('checkpoints/skylar-100M-Chat-v4/best/tokenizer.json')
if tok.decoder is None:
    tok.decoder = decoders.ByteLevel()

msgs = [
    {'role': 'system', 'content': 'Sei un assistente utile.'},
    {'role': 'user', 'content': 'Ciao, come stai?'},
]
ids = encode_chatml(msgs, tok, add_generation_prompt=True)
input_ids = torch.tensor([ids], device='cuda')

# Generate 60 raw token IDs
cur = input_ids.clone()
raw_ids = []
for _ in range(60):
    with torch.no_grad():
        out = model(cur[:, -model.config.max_seq_len:])
    nxt = out['logits'][0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
    cur = torch.cat([cur, nxt], dim=1)
    raw_ids.append(nxt.item())

print('Raw token IDs (first 60):')
for i, tid in enumerate(raw_ids):
    decoded = repr(tok.decode([tid]))
    marker = ' ◀◀◀ IM_END' if tid == 4 else (' ◀ IM_START' if tid == 3 else '')
    print(f'  {i:>3}: ID={tid:>6}  decoded={decoded:<30}{marker}')
