import numpy as np
import os
import torch
from text.symbols import symbols
import models
import utils as glow_utils
from text import text_to_sequence, cmudict


def init(checkpoint_path, config_path, device="cuda"):
    hps = glow_utils.get_hparams_from_json(checkpoint_path, config_path)
    model = models.FlowGenerator(
        len(symbols),
        out_channels=hps.data.n_mel_channels,
        **hps.model).to(device)

    if os.path.isdir(checkpoint_path):
        checkpoint_path = glow_utils.latest_checkpoint_path(checkpoint_path)
    glow_utils.load_checkpoint(checkpoint_path, model)
    model.decoder.store_inverse()  # do not calcuate jacobians for fast decoding
    _ = model.eval()

    cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

    return cmu_dict, model

# function to generate speech
def predict(cmu_dict, model, text, device="cuda"):
  sequence = np.array(text_to_sequence(text, ['english_cleaners'], cmu_dict))[None, :]
  x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
  x_tst_lengths = torch.tensor([x_tst.shape[1]]).to(device)
  with torch.no_grad():
    noise_scale = .667
    length_scale = 1.20
    (y_gen_tst, *r), attn_gen, *_ = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
    return y_gen_tst.cpu()


def repl_test():
    checkpoint_path = './pretrained.pth'
    config_path = './configs/base.json'
    cmu_dict, model = init(checkpoint_path, config_path, device="cpu")

    text = "say something"
    audio = predict(cmu_dict, model, text, device="cpu")