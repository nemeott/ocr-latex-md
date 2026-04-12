enc_out = encoder(x)
dec_out = latex_decoder(enc_out, max_len=128, target_tokens=None)
pred_ids = dec_out.token_ids
