import librosa, soundfile
import numpy as np

import torch
from torch import nn

from viseme_lipsync.model import VisemeModel



SAMPLE_RATE = 16000

#fname = "TTS_Processed_test1.wav"
#fname="test.wav"
#fname="TTS_test2.wav"
fname="TTS_test_vowel.wav"
#fname="TTS_testtest.wav"
audio, sample_rate = librosa.load(fname, sr=SAMPLE_RATE, mono=True)


model=VisemeModel()
model(audio)

"""import matplotlib.pyplot as plt
time = np.linspace(0, len(audio)/sample_rate, len(audio)) # time axis
fig, ax1 = plt.subplots() # plot
ax1.plot(time, audio, color = 'b', label='speech waveform')
plt.show()"""