import matplotlib.pyplot as plt
import numpy as np
import scipy

dodgersCar = np.loadtxt('data/dodgers.cars.data')
dodgersEvents = np.loadtxt('data/dodgers.events.data')

cars = dodgersCar[:,1]
events = dodgersEvents[:,1]

bach882 = np.loadtxt('data/Bach.882.txt')
bach1378 = np.loadtxt('data/Bach.1378.txt')
bach2756 = np.loadtxt('data/Bach.2756.txt')
bach5512 = np.loadtxt('data/Bach.5512.txt')
bach11025 = np.loadtxt('data/Bach.11025.txt')
bach44100 = np.loadtxt('data/Bach.44100.txt')

def _fft_plot(audio, sampling_rate):
    n = len(audio)
    T = 1/sampling_rate
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0,1.0/(2.0*T),n//2)

    return xf, 2.0/n * np.abs(yf[:n//2])


def get_xy(samples, rate):
    x, y = _fft_plot(samples, rate)
    return x, y

fft882 = get_xy(bach882, 882)
fft1378 = get_xy(bach1378, 1378)
fft2756 = get_xy(bach2756, 2756)
fft5512 = get_xy(bach5512, 5512)
fft11025 = get_xy(bach11025, 11025)
fft44100 = get_xy(bach44100, 44100)

# plt.plot(fft882)
# plt.plot(fft1378)
# plt.plot(fft2756)
# plt.plot(fft5512)
# plt.plot(fft11025)

plt.plot(fft44100[0], fft44100[1], label = '44100')
plt.plot(fft11025[0], fft11025[1], label = '11025')
plt.plot(fft5512[0], fft5512[1], label = '5512')
plt.plot(fft2756[0], fft2756[1], label = '2756')
plt.plot(fft1378[0], fft1378[1], label = '1378')
plt.plot(fft882[0], fft882[1], label = '882')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0,7000)
plt.savefig('bach')
plt.show()


dc = get_xy(cars, len(cars))
de = get_xy(events, len(events))

fig1 = plt.figure(1)
plt.plot(dc)
plt.show()