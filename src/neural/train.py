import torch

from neural import LatexAttentionGRUDecoder, OCRHandwrittenCNNEncoder

encoder = OCRHandwrittenCNNEncoder(
    image_height=32,  # Must match input image size
    image_width=256,
    embedding_dim=512,
    dropout=0.1,
)
# print_model_summary(encoder, batch_size=8)

latex_decoder = LatexAttentionGRUDecoder(
    vocab_size=vocab_size_latex,
    encoder_dim=encoder.embedding_dim,
    pad_id=latex_pad_id,
    bos_id=latex_bos_id,
    eos_id=latex_eos_id,
)

x = torch.randn(8, 1, 32, 256)  # images
y_latex = torch.randint(0, vocab_size_latex, (8, 64))  # token ids; ideally y[:,0]=BOS

enc_out = encoder(x)  # (N,S,E)
dec_out = latex_decoder(enc_out, target_tokens=y_latex, teacher_forcing=True)

# typical shift: predict next token
logits = dec_out.logits[:, :-1, :]  # (N,T-1,V)
targets = y_latex[:, 1:]  # (N,T-1)
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=latex_pad_id,
)
loss.backward()
