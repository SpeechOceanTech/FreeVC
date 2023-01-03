import os
import re
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import random

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="configs/freevc.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="checkpoints/freevc.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="convert.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/freevc", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    parser.add_argument("--aishell_root_dir", default="/data/duhu/corpus/zh-cn/aishell")
    parser.add_argument("--tgt_data_dir", default="/data/duhu/corpus/King-ASR-783/")
    args = parser.parse_args()
    
    src_files = Path(args.aishell_root_dir).rglob("*.wav")
    src_files = [str(ele) for ele in src_files]
    src_spks2wav = defaultdict(list)
    for src_file in src_files:
        spk = re.findall("S[0-9]{4}", src_file)[0]
        src_spks2wav[spk].append(src_file)
    
    tgt_files = Path(args.tgt_data_dir).rglob("*.WAV")
    tgt_files = [str(ele) for ele in tgt_files]
    
    tgt_spks2wav = defaultdict(list)
    tgt_spks = []
    for tgt_file in tgt_files:
        spk = re.findall("SPEAKER[0-9]+", tgt_file)[0]
        tgt_spks2wav[spk].append(tgt_file)
    
    tgt_spks = list(tgt_spks2wav.keys())

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    # with open(args.txtpath, "r") as f:
    #     for rawline in f.readlines():
    #         title, src, tgt = rawline.strip().split("|")
    #         titles.append(title)
    #         srcs.append(src)
    #         tgts.append(tgt)
    r_code = "r4"
    with open(f"aishell_vc_{r_code}.log", 'w', encoding='utf-8') as f:        
        for spk, src_files in src_spks2wav.items():
            for src_file in src_files:
                title = src_file.replace("/aishell/", f"/aishell_vc_{r_code}/")
                name = os.path.basename(title)[:-4] + ".wav"
                title = os.path.join(os.path.dirname(title), name)
                os.makedirs(os.path.dirname(title), exist_ok=True)
                titles.append(title)
                srcs.append(src_file)
                tgt_spk = random.sample(tgt_spks, 1)[0]
                tgt_file = random.sample(tgt_spks2wav[tgt_spk], 1)[0]
                tgts.append(tgt_file)
                f.write(f"{title}\t{src_file}\t{tgt_file}\n")
           

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt, 
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            c = utils.get_content(cmodel, wav_src)
            
            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            write(title, hps.data.sampling_rate, audio)
            
