import numpy as np
import os
import torch
from glow_tts.text.symbols import symbols
from glow_tts import models
import glow_tts.utils as glow_utils
from glow_tts.text import text_to_sequence, cmudict
import glow_tts.commons as commons


def init(checkpoint_path, config_path, device="cpu"):
    hps = glow_utils.get_hparams_from_file(config_path)
    model = models.FlowGenerator(
        len(symbols) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        **hps.model).to(device)

    if os.path.isdir(checkpoint_path):
        checkpoint_path = glow_utils.latest_checkpoint_path(checkpoint_path)
    glow_utils.load_checkpoint(checkpoint_path, model)
    model.decoder.store_inverse()  # do not calcuate jacobians for fast decoding
    _ = model.eval()

    cmu_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/cmu_dictionary')
    cmu_dict = cmudict.CMUDict(cmu_dict_path)

    return cmu_dict, model


# function to generate speech
def predict(cmu_dict, model, text, device="cpu", length_scale=1.0, blanks=False):
    if blanks:
        text_norm = text_to_sequence(text.strip(), ['english_cleaners'], cmu_dict)
        text_norm = commons.intersperse(text_norm, len(symbols))
    else:  # adding spaces at the beginning and the end of utterance improves quality
        tst_stn = " " + text.strip() + " "
        text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)

    sequence = np.array(text_norm)[None, :]
    print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).to(device)
    with torch.no_grad():
        noise_scale = .667
        (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale,
                                                    length_scale=length_scale)
        return y_gen_tst.cpu()


def repl_test():

    # blanks
    blanks = True
    checkpoint_path = 'data/pretrained_blank.pth'
    checkpoint_path = 'data/G_3009.pth'
    config_path = 'data/blanks.json'

    # normal
    blanks = False
    checkpoint_path = 'data/pretrained.pth'
    config_path = 'glow_tts/configs/base.json'

    # load model
    device = "cpu"
    cmu_dict, model = init(checkpoint_path, config_path, device=device)

    # test prediction
    text = "what the actual fuck is going on?"
    audio = predict(cmu_dict, model, text, device=device, blanks=blanks)
