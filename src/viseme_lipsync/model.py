import numpy as np
from pathlib import Path

import torch
from torch import nn

from conformer_ppg.model import PPGPhonemeModel
from viseme_lipsync.util import *
phoneme_viseme_dict={
    'B': ['음'],
    'F': ['음'],
    'M': ['음'],
    'P': ['음'],
    'V': ['음']}
phoneme_vowel_viseme_dict = {
    'SIL': ['SIL'],
    'AA0': ['ㅏ'],
    'AA1': ['ㅏ'],
    'AA2': ['ㅏ'],
    'AE0': ['ㅔ'],
    'AE1': ['ㅔ'],
    'AE2': ['ㅔ'],
    'AH0': ['ㅓ'],
    'AH1': ['ㅓ'],
    'AH2': ['ㅓ'],
    'AO0': ['ㅗ'],
    'AO1': ['ㅗ'],
    'AO2': ['ㅗ'],
    'AW0': ['ㅏ','ㅜ'],
    'AW1': ['ㅏ','ㅜ'],
    'AW2': ['ㅏ','ㅜ'],
    'AY0': ['ㅏ','ㅣ'],
    'AY1': ['ㅏ','ㅣ'],
    'AY2': ['ㅏ','ㅣ'],
    'EH0': ['ㅓ'],
    'EH1': ['ㅓ'],
    'EH2': ['ㅓ'],
    'ER0': ['ㅓ'],
    'ER1': ['ㅓ'],
    'ER2': ['ㅓ'],
    'EY0': ['ㅔ'],
    'EY1': ['ㅔ'],
    'EY2': ['ㅔ'],
    'IH0': ['ㅣ'],
    'IH1': ['ㅣ'],
    'IH2': ['ㅣ'],
    'IY0': ['ㅣ'],
    'IY1': ['ㅣ'],
    'IY2': ['ㅣ'],
    'OW0': ['ㅗ','ㅜ'],
    'OW1': ['ㅗ','ㅜ'],
    'OW2': ['ㅗ','ㅜ'],
    'OY0': ['ㅗ','ㅣ'],
    'OY1': ['ㅗ','ㅣ'],
    'OY2': ['ㅗ','ㅣ'],
    'UH0': ['ㅜ'],
    'UH1': ['ㅜ'],
    'UH2': ['ㅜ'],
    'UW0': ['ㅜ'],
    'UW1': ['ㅜ'],
    'UW2': ['ㅜ'],
    'W': ['ㅜ','ㅣ'],
    'Y': ['ㅜ']}

phoneme_dict = {
    'SIL': ['SIL'],
    'AA0': ['ㅏ'],
    'AA1': ['ㅏ'],
    'AA2': ['ㅏ'],
    'AE0': ['ㅔ'],
    'AE1': ['ㅔ'],
    'AE2': ['ㅔ'],
    'AH0': ['ㅓ'],
    'AH1': ['ㅓ'],
    'AH2': ['ㅓ'],
    'AO0': ['ㅗ'],
    'AO1': ['ㅗ'],
    'AO2': ['ㅗ'],
    'AW0': ['ㅏ','ㅜ'],
    'AW1': ['ㅏ','ㅜ'],
    'AW2': ['ㅏ','ㅜ'],
    'AY0': ['ㅏ','ㅣ'],
    'AY1': ['ㅏ','ㅣ'],
    'AY2': ['ㅏ','ㅣ'],
    'EH0': ['ㅓ'],
    'EH1': ['ㅓ'],
    'EH2': ['ㅓ'],
    'ER0': ['ㅓ'],
    'ER1': ['ㅓ'],
    'ER2': ['ㅓ'],
    'EY0': ['ㅔ'],
    'EY1': ['ㅔ'],
    'EY2': ['ㅔ'],
    'IH0': ['ㅣ'],
    'IH1': ['ㅣ'],
    'IH2': ['ㅣ'],
    'IY0': ['ㅣ'],
    'IY1': ['ㅣ'],
    'IY2': ['ㅣ'],
    'OW0': ['ㅗ','ㅜ'],
    'OW1': ['ㅗ','ㅜ'],
    'OW2': ['ㅗ','ㅜ'],
    'OY0': ['ㅗ','ㅣ'],
    'OY1': ['ㅗ','ㅣ'],
    'OY2': ['ㅗ','ㅣ'],
    'UH0': ['ㅜ'],
    'UH1': ['ㅜ'],
    'UH2': ['ㅜ'],
    'UW0': ['ㅜ'],
    'UW1': ['ㅜ'],
    'UW2': ['ㅜ'],
    'W': ['ㅜ','ㅣ'],
    'Y': ['ㅜ'],								
    'B': ['ㅂ'],
    'F': ['ㅍ'],
    'M': ['ㅁ'],
    'N': ['ㄴ'],
    'P': ['ㅍ'],
    'V': ['ㅂ'],
    'D': ['ㄷ'],
    'DH': ['ㄸ'],
    'L': ['ㄹ'],
    'R': ['ㄹ'],
    'S': ['ㅅ'],
    'SH': ['ㅅ'],
    'T': ['ㅌ'],
    'TH': ['ㄸ'],
    'CH': ['ㅊ'],
    'JH': ['ㅈ'],
    'Z': ['ㅈ'],
    'ZH': ['ㅈ'],
    'G': ['ㄱ'],
    'K': ['ㅋ'],
    'HH': ['ㅎ']
}                               

class VisemeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.SAMPLE_RATE = 16000
        self.phoneme_len = 0.01
        self.chunk = 160

    def get_active_phoneme(self,audio):
        audio_length = len(audio)
        audio = audio.reshape(1, audio_length)
        input_audio = torch.tensor(audio)

        ppg_model = PPGPhonemeModel.from_static()
        output = ppg_model(input_audio)
        output = output.numpy()[0]

        # output phoneme post-processing
        output_len = output.shape[0]

        active_phoneme = np.zeros(shape=(output.shape))
        for i in range(output_len):
            idx = np.argmax(output[i])
            active_phoneme[i, idx] = 1

        # Phoneme print
        active_phoneme_list = []
        librispeech_path=Path(__file__).parent.joinpath("librispeech_phoneme.txt")
        phoneme_list=get_libri_phoneme(librispeech_path)
        
        for i in range(output_len):
            idx = np.argmax(active_phoneme[i])
            active_phoneme_list.append(phoneme_list[idx])

        return active_phoneme_list
    
    def get_accent(self,audio):

        """
        ###timming###
        audio shape = 111467
        111467 / 16000 = 6.966s
        6.966/694 = 0.01s

        한 phoneme당 0.01초

        ###accent###
        SAMPLE_RATE = 16000
        0.01 초에 160

        160당 절대값의 평균 리스트
        """
        chunk = self.chunk

        accent = []
        while len(audio) > chunk:
            arr = np.split(audio, [chunk])
            accent.append(np.mean(abs(arr[0])))
            audio = arr[1]
        accent.append(np.mean(abs(audio)))

        return accent   
    
    def phonme_to_viseme(self,phoneme,accent,timming):
        viseme_list = []
        timming_list=[]
        accent_list=[]
        for i in range(0,len(phoneme)):
            accent_list.append(accent[i])
            timming_list.append(timming[i])
            if phoneme[i] in phoneme_vowel_viseme_dict:
                viseme_list.extend(phoneme_vowel_viseme_dict[phoneme[i]])
                if len(phoneme_vowel_viseme_dict[phoneme[i]])>1:
                    timming_list[-1]=timming_list[-1]/2
                    timming_list.append(timming[i]/2)
                    accent_list.append(accent[i])
            elif phoneme[i] in phoneme_viseme_dict:
                viseme_list.extend(phoneme_viseme_dict[phoneme[i]])   
            else:
                viseme_list.append(phoneme[i])
        return viseme_list,accent_list,timming_list
    
    def phonme_to_krviseme(self,phoneme):
        viseme_kr_list = []
        for ph in phoneme:
            if ph in phoneme_dict:
                viseme_kr_list.extend(phoneme_dict[ph])
            else:
                viseme_kr_list.append(ph)
        return viseme_kr_list
    
    def phonme(self,phoneme,accent,phoneme_len):
        phoneme_list=[phoneme[0]]
        timming_list=[0]
        accent_list=[accent[0]]
        for i in range(1,len(phoneme)):
            if phoneme[i] == phoneme_list[-1]:
                timming_list[-1]+=phoneme_len
                accent_list[-1]=(accent_list[-1]+accent[i])/2
            else:
                phoneme_list.append(phoneme[i])
                timming_list.append(phoneme_len)
                accent_list.append(accent[i])
        return phoneme_list,accent_list,timming_list

    def forward(self,audio,SR=16000):
        self.SAMPLE_RATE = SR
        wav_length=len(audio)/SR

        active_phoneme_list = self.get_active_phoneme(audio)
        phoneme_len=wav_length/len(active_phoneme_list)
        self.phoneme_len = phoneme_len  
        self.chunk= round(phoneme_len * SR)

        accent= self.get_accent(audio)
        phoneme_list,accent_list, timming_list=self.phonme(active_phoneme_list,accent,phoneme_len)

        viseme_list,accent_list,timming_list=self.phonme_to_viseme(phoneme_list,accent_list,timming_list)

        print("## active_phoneme_list:", active_phoneme_list)
        print("## active_phoneme_list:", len(active_phoneme_list))

        print("## phoneme_list:", phoneme_list)
        print("## phoneme_list:", len(phoneme_list))
        print("## accent_list:", accent_list)
        print("## accent_list:", len(accent_list))
        print("## timing_list:", timming_list)
        print("## timing_list:", len(timming_list))

        print("## viseme_list:", viseme_list)
        print("## viseme_list:", len(viseme_list))

        print("## phoneme_len:", self.phonme_to_krviseme(phoneme_list))
        print("## phoneme_len:", len(self.phonme_to_krviseme(phoneme_list)))
        print(sum(timming_list))

        return active_phoneme_list
    
