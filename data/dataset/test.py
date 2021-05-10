#
# waveform, sample_rate = librosa.load('/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/genres_original/blues/blues.00000.wav', sr=16000)
# # waveform, sample_rate = torchaudio.load(,)
# print(sample_rate)
# mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels = 128,
#                      n_fft = 2048, hop_length = 512,)(waveform)
# print(mel_specgram.size())
# import numpy as np
# ones = np.ones((2,2))
# print(ones)
# p = np.pad(ones,pad_width=constant_values=0)
# print(p)
