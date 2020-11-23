#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from math import pi as PI
from math import sin
from scipy import signal
import IPython.display as ipd



# In[2]:
# <b>Zadanie 3.1<b>

N = 64

x = np.arange(N)

y = np.sin((x/N) * 2 * PI)
Y = np.fft.fft(y) * 2 / N

fig, plots = plt.subplots(3, 2, figsize= (12,12), dpi= 100)

for subPlots in plots:
    for plot in subPlots:
        plot.grid(True)
        plot.set_xlabel("Nr Pasma Częstotliwościowego")
        plot.set_ylabel("Amplituda")

plots[0][0].stem(np.real(y))
plots[0][0].set_title("Re(y)")
plots[0][0].set_xlabel("Nr Próbki")
plots[0][1].stem(np.imag(y))
plots[0][1].set_title("Im(y)")
plots[0][1].set_xlabel("Nr Próbki")

plots[1][0].stem(np.real(Y))
plots[1][0].set_title("Re(FFT(y))")
plots[1][0].set_ylim(-1, 1)
plots[1][1].stem(np.imag(Y))
plots[1][1].set_title("Im(FFT(y))")

plots[2][0].stem(np.absolute(Y))
plots[2][0].set_title("|FFT(y)|")
plots[2][0].set_ylabel("Magnituda")
plots[2][1].stem(np.angle(Y)/PI)
plots[2][1].set_title("Phase(FFT(y))")
plots[2][1].set_ylabel("Faza [rad*PI]")

plt.subplots_adjust(hspace=0.4)
plt.savefig("Zadanie_3_1\\plots.png")



# In[3]:
# <b>Zadanie 3.2<b>

N = 64
x = np.arange(N)

y = []

y.append(np.cos(PI*(2*x/N)))
y.append(0.5 * np.cos(PI*(4*x/N)))
y.append(0.25 * np.cos(PI*(8*x/N + .5)))
y.append(y[0] + y[1] + y[2])

plt.figure(figsize= (12,8), dpi= 200)
line1 = plt.plot(x, y[0], color = "red", label = "y1 = cos(PI*(2*x/N + .25))")
plt.plot(x, y[1], color = "green", label = "y2 = 0.5 * cos(PI*(4*x/N))")
plt.plot(x, y[2], color = "blue", label = "y3 = 0.25 * cos(PI*(8*x/N + .5))")
plt.plot(x, y[3], color = "yellow", label = "y4 = y1+y2+y3")
plt.title("Wykresy różnych sygnałów sinusoidalnych i ich sumy.")

plt.legend()

plt.grid(True)

plt.savefig("Zadanie_3_2\\1.png")


fig, plots = plt.subplots(4, 2, figsize= (6,12), dpi= 100)

Y = []
for i in range(4):
    Y.append(np.fft.fft(y[i]) * 2 / N)
    plots[i][0].stem(x, np.absolute(Y[i]))
    plots[i][0].set_title(f"|FFT(y{i+1})|")
    plots[i][0].set_xlabel("Numer pasma częstotliwościowego")
    plots[i][0].set_ylabel("Magnituda")
    plots[i][0].set_ylim(-1.1, 1.1)

for i in range(4):
    Y.append(np.fft.fft(y[i]) * 2 / N)
    plots[i][1].stem(x, np.angle(Y[i])/PI)
    plots[i][1].set_title(f"Phase(FFT(y{i+1}))")
    plots[i][1].set_xlabel("Numer pasma częstotliwościowego")
    plots[i][1].set_ylabel("Faza [rad*PI]")

plt.suptitle("\nAnaliza widma sygnałów z poprzednich wykresów.")
plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.savefig("Zadanie_3_2\\2.png")


# In[4]:
# <b>Zadanie 3.3.1<b>

N = 64

x = np.arange(N)

y = np.sin((x/N) * 2 * PI)

fig, plots = plt.subplots(3, 2, figsize= (12,12), dpi= 100)

for subPlots in plots:
    for plot in subPlots:
        plot.grid(True)
        plot.set_xlabel("Nr Pasma Częstotliwościowego")
        plot.set_ylabel("Amplituda")

plots[0][0].stem(np.real(y))
plots[0][0].set_title("Re(y)")
plots[0][0].set_xlabel("Nr Próbki")
plots[0][1].stem(np.imag(y))
plots[0][1].set_title("Im(y)")
plots[0][1].set_xlabel("Nr Próbki")

Y = np.fft.fft(y)
plots[1][0].stem(np.absolute(Y))
plots[1][0].set_title("|FFT(y)|")
plots[1][0].set_ylabel("Magnituda")
plots[1][1].stem(np.angle(Y)/PI)
plots[1][1].set_title("Phase(FFT(y))")
plots[1][1].set_ylabel("Faza [rad*PI]")

plots[2][0].stem(np.real(np.fft.ifft(Y)))
plots[2][0].set_title("Re(IFFT(FFT(y)))")
plots[2][0].set_ylim(-1, 1)
plots[2][1].stem(np.imag(np.fft.ifft(Y)))
plots[2][1].set_title("Im(IFFT(FFT(y)))")
plots[2][1].set_ylim(-1.1, 1.1)

plt.suptitle("\nOdzyskanie sygnału z transformaty Fouriera")
plt.subplots_adjust(hspace=0.4)
plt.savefig("Zadanie_3_3\\1.png")

# In[12]:

<b> Huwdu <b>

# In[5]:
# <b>Zadanie 3.3.2<b>

N = 64
x = np.arange(N)

y = []

y.append(np.cos(PI*(2*x/N + .25)))
y.append(0.5 * np.cos(PI*(4*x/N)))
y.append(0.25 * np.cos(PI*(8*x/N + .5)))
sum = y[0] + y[1] + y[2]

fig, plots = plt.subplots(4, 1, figsize= (6,12), dpi= 100)

plots[0].stem(sum)
plots[0].set_title("y")
plots[0].set_xlabel("Numer Próbki")
plots[0].set_ylabel("Amplituda")
Y = np.fft.fft(sum)
plots[1].stem(np.absolute(Y))
plots[1].set_title("|FFT(y)|")
plots[1].set_xlabel("Numer Pasma Częstotliwościowego")
plots[1].set_ylabel("Magnituda")
plots[2].stem(np.angle(Y))
plots[2].set_title("Phase(FFT(y))")
plots[2].set_xlabel("Numer Pasma Częstotliwościowego")
plots[2].set_ylabel("Faza [rad*PI]")
plots[3].stem(np.fft.ifft(Y))
plots[3].set_title("IFFT(FFT(y))")
plots[3].set_xlabel("Numer Próbki")
plots[3].set_ylabel("Amplituda")

plt.suptitle("\nOdzyskanie sygnału z transformaty Fouriera")
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("Zadanie_3_3\\2.png")


# In[6]:
# <b>Zadanie 3.4<b>
from math import sin
from math import cos

N = 32

x = np.arange(N)
x = (x/N)*2*PI
y = []
for i in range(N):
    y.append(sin(x[i]) + cos(x[i]) * 1j)


fig, plots = plt.subplots(3, 2, figsize= (12,12), dpi= 100)

for subPlots in plots:
    for plot in subPlots:
        plot.grid(True)
        plot.set_xlabel("Nr Pasma Częstotliwościowego")
        plot.set_ylabel("Amplituda")

plots[0][0].stem(np.real(y))
plots[0][0].set_title("Re(y)")
plots[0][0].set_xlabel("Nr Próbki")
plots[0][1].stem(np.imag(y))
plots[0][1].set_title("Im(y)")
plots[0][1].set_xlabel("Nr Próbki")


Y = np.fft.fft(y) * 2 / N

plots[1][0].stem(np.real(Y))
plots[1][0].set_title("Re(FFT(y))")
plots[1][0].set_ylim(-1, 1)
plots[1][1].stem(np.imag(Y))
plots[1][1].set_title("Im(FFT(y))")

plots[2][0].stem(np.absolute(Y))
plots[2][0].set_title("|FFT(y)|")
plots[2][0].set_ylabel("Magnituda")
plots[2][1].stem(np.angle(Y)/PI)
plots[2][1].set_title("Phase(FFT(y))")
plots[2][1].set_ylabel("Faza [rad*PI]")

plt.suptitle("Widmo sygnału zespolonego", fontsize= 'xx-large')
plt.subplots_adjust(hspace=0.4)
plt.savefig("Zadanie_3_4\\1.png")



# In[7]:
# <b>Zadanie 3.5<b>

from scipy.linalg import dft

def MyDFT(y):
    return dft(np.size(y)) @ y

def MyIDFT(Y):
    return dft(np.size(y)) @ -Y

N = 64

x = np.arange(N)

y = np.sin((x/N) * 2 * PI)
Y = MyDFT(y)
# Y = MyIDFT(Y)

#region >> Plots <<
fig, plots = plt.subplots(3, 2, figsize= (12,12), dpi= 100)
fig.delaxes(plots[0][1])

for subPlots in plots:
    for plot in subPlots:
        plot.grid(True)
        plot.set_xlabel("Nr Pasma Częstotliwościowego")

plots[0][0].stem(y)
plots[0][0].set_title("y")
plots[0][0].set_xlabel("Nr Próbki")
plots[0][0].set_ylabel("Amplituda")

plots[1][0].stem(np.absolute(np.fft.fft(y)))
plots[1][0].set_title("|FFT(y)|")
plots[1][0].set_ylabel("Magnituda")
plots[1][1].stem(np.absolute(Y))
plots[1][1].set_title("|MyFFT(y)|")
plots[1][1].set_ylabel("Magnituda")

plots[2][0].stem(np.angle(np.fft.fft(y)))
plots[2][0].set_title("Phase(FFT(y))")
plots[2][0].set_ylabel("Faza [rad]")
plots[2][1].stem(np.angle(Y))
plots[2][1].set_title("Phase(MyFFT(y))")
plots[2][1].set_ylabel("Faza [rad]")

plt.subplots_adjust(hspace=0.4)
plt.savefig("Zadanie_3_5\\myPlots.png")

#endregion



# In[8]:
# <b>Zadanie 3.6<b>

N = 32

x = np.arange(N)

y = np.sin((x/N) * 2 * PI * 2.5)

windows = []
windows.append([signal.windows.bartlett(N), "red", "Bartlett"])
windows.append([signal.windows.blackman(N), "yellow", "Blackman"])
windows.append([signal.windows.hamming(N), "black", "Hamming"])
windows.append([signal.windows.hann(N), "green", "Hann"])
windows.append([signal.windows.kaiser(N, 1), "violet", "Kaiser"])
windows.append([signal.windows.boxcar(N), "cyan", "Rectangular"])


Y = []
Y.append(np.fft.fft(y))
for window in windows:
    Y.append(y * window[0])


#region >> Plots <<
fig, plots = plt.subplots(3, 3, figsize= (12,9), dpi= 100)

for subPlots in plots:
    for plot in subPlots:
        plot.grid(True)
        plot.set_xlabel("Nr Pasma Częstotliwościowego")
        plot.set_ylabel("Magnituda")

plots[0][0].stem(y)
plots[0][0].set_title("Sygnał")
plots[0][0].set_xlabel("Nr Próbki")
plots[0][0].set_xlabel("Amplituda")
for window in windows:
    plots[0][1].plot(window[0], color=window[1], label=window[2])
plots[0][1].set_title("Funkcje okna")
plots[0][1].set_xlabel("Nr Próbki")
plots[0][1].set_ylabel("Waga")
plots[0][1].legend(loc='lower center', fontsize='xx-small')
plots[0][2].stem(np.absolute(Y[0]))
plots[0][2].set_title("|FFT(Sygnał)|")
for y in range(2):
    for x in range(3):
        plots[y+1][x].stem(np.absolute(np.fft.fft(Y[1+x+y*3])))
        plots[y+1][x].set_title(f"|FFT(Sygnał*{windows[x+y*3][2]})|")

plt.suptitle("\nFunkcje okna.", fontsize='xx-large')
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.savefig("Zadanie_3_6\\Plots.png")
#endregion

# %%
