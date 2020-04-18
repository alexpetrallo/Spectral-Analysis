import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display
import scipy
from scipy.signal import argrelextrema

filepath = 'data/guitar.wav'
eMajPath = 'data/myData/eMaj.wav'
eNotePath = 'data/myData/eNote.wav'
eHarmPath = 'data/myData/eHarm.wav'
bHarmPath = 'data/myData/bHarm.wav'
gHarmPath = 'data/myData/gHarm.wav'
aSharp = 'data/myData/a#.wav'
tritonePath = 'data/myData/tritone.wav'

ePath = 'data/myData/e.wav'
gShPath = 'data/myData/g#.wav'
dPath = 'data/myData/d.wav'
gPath = 'data/myData/g.wav'
beatsPath = 'data/myData/beats.wav'
eOutPath = 'data/myData/eOut.wav'



data = np.loadtxt('data/guitar.txt')

samples, sampling_rate = librosa.load(filepath, sr = None, mono=True, offset = 0.0, duration = None)
eMajSamples, eMaj_sampling_rate = librosa.load(eMajPath, sr = None, mono=True, offset = 0.0, duration = None)


eHsamples, eH_sampling_rate = librosa.load(eHarmPath, sr = None, mono=True, offset = 0.0, duration = None)
bHsamples, bH_sampling_rate = librosa.load(bHarmPath, sr = None, mono=True, offset = 0.0, duration = None)
gHsamples, gH_sampling_rate = librosa.load(gHarmPath, sr = None, mono=True, offset = 0.0, duration = None)

aSharpsamples, aS_sampling_rate = librosa.load(aSharp, sr = None, mono=True, offset = 0.0, duration = None)
tritoneSamples, tritone_sampling_rate = librosa.load(tritonePath, sr = None, mono=True, offset = 0.0, duration = None)

eSamples, e_sampling_rate = librosa.load(ePath, sr = None, mono=True, offset = 0.0, duration = None)
gShSamples, gSh_sampling_rate = librosa.load(gShPath, sr = None, mono=True, offset = 0.0, duration = None)
dSamples, d_sampling_rate = librosa.load(dPath, sr = None, mono=True, offset = 0.0, duration = None)
gSamples, g_sampling_rate = librosa.load(gPath, sr = None, mono=True, offset = 0.0, duration = None)
beatsSamples, beats_sampling_rate = librosa.load(beatsPath, sr = None, mono=True, offset = 0.0, duration = None)
outSamples, out_sampling_rate = librosa.load(eOutPath, sr = None, mono=True, offset = 0.0, duration = None)


freqNoteDict = {}
file = open('data/myData/noteFreqData.txt', 'r')
f = file.readlines()
for l in f:
    (val, key) = l.split()
    freqNoteDict[float(key)] = val


print(len(samples))
print(sampling_rate)

def _fft_plot(audio, sampling_rate):
    n = len(audio)
    T = 1/sampling_rate
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0,1.0/(2.0*T),n//2)
    return xf, 2.0/n * np.abs(yf[:n//2])

def get_xy(samples, rate):
    x, y = _fft_plot(samples, rate)
    return x, y


# eMajTones = argrelextrema(y1, np.greater)

def get_freqs(x,y):
    maxIndeces = []
    cutY = []
    cutX = []
    freq = []
    mag = []

    for e in range(len(y)):
        if y[e] > .005:  # and x[e] < 2000:
            maxIndeces.append(e)
            cutY.append(y[e])
            cutX.append(x[e])

    cutY = np.array(cutY)
    tones = argrelextrema(cutY, np.greater)

    for e in tones[0]:
        freq.append(cutX[e])
        mag.append(cutY[e])

    return freq, mag


def get_notes(freqs, mags):
    noteList = []
    for note in freqs:
        noteList.append(freqNoteDict.get(note, freqNoteDict[min(freqNoteDict.keys(), key=lambda k: abs(k - note))]))

    noteMags = [(noteList[i], mags[i]) for i in range(0, len(noteList))]

    return noteMags

x, y = get_xy(samples, sampling_rate)
x1, y1 = get_xy(eMajSamples, eMaj_sampling_rate)
xH, yH = get_xy(eHsamples, eH_sampling_rate)

xBh, yBh = get_xy(bHsamples, bH_sampling_rate)
xGh, yGh = get_xy(gHsamples, gH_sampling_rate)

xAs, yAs = get_xy(aSharpsamples, aS_sampling_rate)
xTri, yTri = get_xy(tritoneSamples, tritone_sampling_rate)

eX, eY = get_xy(eSamples, e_sampling_rate)
gShX, gShY = get_xy(gShSamples, gSh_sampling_rate)
dX, dY = get_xy(dSamples, d_sampling_rate)
gX, gY = get_xy(gSamples, g_sampling_rate)

beatsX, beatsY = get_xy(beatsSamples, beats_sampling_rate)
outX, outY = get_xy(outSamples, out_sampling_rate)





eNoteFreq, eMags = get_freqs(x,y)
eNotesMags = get_notes(eNoteFreq, eMags)

eMajNoteFreq, eMajMags = get_freqs(x1,y1)
eMajNotesMags = get_notes(eMajNoteFreq, eMajMags)

eHarmF, eHarmM = get_freqs(xH, yH)
eHarmNotesMags = get_notes(eHarmF, eHarmM)

tritoneF, tritoneM = get_freqs(xAs,yAs)
tritoneNoteMag = get_notes(tritoneF, tritoneM)



print(eNotesMags)
print(eMajNotesMags)
print(eHarmNotesMags)
print(tritoneNoteMag)




##Plots
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
#Harmonic Series Notes
ax.text(65, .03, 'E')
ax.text(130, .055, 'E')
ax.text(235, .033, 'B')
ax.text(327, .018, 'E')
ax.text(404, .01, 'G#')
ax.text(490, .004, 'B')
ax.text(550, .004, 'Db-D')
ax.text(653, .002, 'E')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
# plt.plot(x,y)
plt.plot(x, y)
plt.grid()
plt.xlim(0, 800)
plt.title('E Note Frequency vs. Magnitude')
fig1.savefig('eNote')
# plt.show()

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
ax.text(65, .024, 'E')
ax.text(130, .055, 'E')
ax.text(235, .033, 'B')
ax.text(320, .012, 'E')
ax.text(365, .012, 'F#')
ax.text(404, .01, 'G#')
ax.text(490, .004, 'B')
ax.text(550, .004, 'Db-D')
ax.text(653, .002, 'E')
#Add major chord notes
ax.text(115, .045, 'B', c='b')
ax.text(185, .018, 'G#', c='b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
# plt.plot(x,y)
plt.plot(x1, y1)
plt.grid()
plt.xlim(0, 800)
plt.title('E Major Chord Frequency vs. Magnitude')
fig2.savefig('eTriad')
# plt.show()

fig3 = plt.figure(3)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
# plt.plot(x,y)
# plt.plot(x1, y1)
plt.plot(xH, yH)
plt.grid()
plt.xlim(0, 800)
plt.title('E Harmonic Frequency vs. Magnitude')
fig3.savefig('eHarm')
# plt.show()

fig4 = plt.figure(4)
ax = fig4.add_subplot(111)
# ax.text(65, .03, 'E')
ax.text(130, .12, 'E')
ax.text(235, .085, 'B')
# ax.text(327, .018, 'E')
ax.text(404, .055, 'G#')
# ax.text(490, .004, 'B')
# ax.text(550, .004, 'Db-D')
# ax.text(653, .002, 'E')

# ax.text(115, .045, 'B', c='b')
# ax.text(185, .018, 'G#', c='b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.plot(xBh,yBh)
plt.plot(xGh, yGh)
plt.plot(xH, yH)
plt.grid()
plt.xlim(0, 800)
plt.title('Major Triad Harmonics Frequency vs. Magnitude')
fig4.savefig('triHarm')
# plt.show()

fig5 = plt.figure(5)
ax = fig5.add_subplot(111)
ax.text(65, .03, 'E')
ax.text(130, .055, 'E')
ax.text(235, .033, 'B')
ax.text(327, .018, 'E')
ax.text(404, .01, 'G#')
ax.text(490, .004, 'B')
ax.text(550, .004, 'Db-D')
ax.text(653, .002, 'E')

ax.text(110,.06, 'A#')
ax.text(230,.045, 'A#')
ax.text(345,.034, 'F')
ax.text(449,.007, 'A#')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.plot(x,y)

plt.plot(xAs,yAs)
# plt.plot(xTri, yTri)
plt.grid()
plt.xlim(0, 800)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Tritone Frequency vs. Magnitude')
fig5.savefig('tritone')
# plt.show()

fig6 = plt.figure(6)
ax = fig6.add_subplot(111)
plt.plot(eX, eY)
plt.plot(gShX, gShY)
plt.plot(dX, dY)
plt.plot(gX, gY)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 800)
plt.title('e7#9 chord Frequency vs. Magnitude')
fig6.savefig('e7#9')
# plt.show()

fig7 = plt.figure(7)
ax = fig7.add_subplot(111)
# plt.plot(beatsX, beatsY)
plt.plot(x,y)
plt.plot(outX, outY)
plt.grid()
plt.xlim(0, 800)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Beats Frequency vs. Magnitude')
fig7.savefig('beats')
plt.show()



# https://pages.mtu.edu/~suits/notefreqs.html
# https://www.youtube.com/watch?v=0lmS5lQ5MSU