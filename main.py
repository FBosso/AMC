#import section
import xarray as xr

#load dataset
dataset = xr.open_dataset("archive/GOLD_XYZ_OSC.0001_1024.hdf5", engine='h5netcdf')

#select the first sample for visualization
X = dataset.X[0]
Y = dataset.Y[0]
Z = dataset.Z[0]


import h5py
import numpy as np
import matplotlib.pyplot as plt

# Carica il file HDF5
filename = 'archive/GOLD_XYZ_OSC.0001_1024.hdf5'
with h5py.File(filename, 'r') as f:
    # Visualizza i gruppi disponibili
    print("Chiavi nel file:", list(f.keys()))  # ['X', 'Y', 'Z']
    
    # Seleziona il primo frame
    frame = f['X'][0]       # shape: (1024, 2)
    label = f['Y'][0]       # one-hot vector (es. [0, 0, ..., 1, ..., 0])
    snr = f['Z'][0]         # SNR (es. -20)

# Separazione delle componenti I e Q
I = frame[:, 0]
Q = frame[:, 1]

# Costruzione del segnale complesso
signal = I + 1j * Q

# Visualizzazione nel dominio del tempo
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(I, label='In-phase (I)')
plt.plot(Q, label='Quadrature (Q)', alpha=0.7)
plt.title('Componente I e Q nel tempo')
plt.xlabel('Campione')
plt.ylabel('Ampiezza')
plt.legend()

# Visualizzazione nel piano complesso (costellazione)
plt.subplot(1, 2, 2)
plt.plot(I, Q, '.', alpha=0.5)
plt.title('Costellazione I/Q (Piano Complesso)')
plt.xlabel('I')
plt.ylabel('Q')
plt.axis('equal')

plt.tight_layout()
plt.show()


modulation_names = [
    'OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32',
    'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64', 'QAM128', 'QAM256',
    'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC', 'AM_DSB_SC', 'FM', 'GMSK', 'OQPS'
]


modulation_index = np.argmax(label)
modulation_name = modulation_names[modulation_index]
print(f"Modulation: {modulation_name}, SNR: {snr} dB")


# Segnale complesso
signal = I + 1j * Q

# Calcola ampiezza e fase
amplitude = np.abs(signal)
phase = np.angle(signal)

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(amplitude)
plt.title(f'Ampiezza del segnale complesso - {modulation_name} @ {snr} dB')
plt.xlabel('Campione')
plt.ylabel('Ampiezza')

plt.subplot(2, 1, 2)
plt.plot(phase)
plt.title('Fase del segnale complesso')
plt.xlabel('Campione')
plt.ylabel('Fase (rad)')

plt.tight_layout()
plt.show()





import h5py
import numpy as np
import matplotlib.pyplot as plt

# === Parametri di esempio ===
fs = 1e6     # frequenza di campionamento: 1 MHz
fc = 100e3   # frequenza della portante: 100 kHz

# === Caricamento frame ===

with h5py.File(filename, 'r') as f:
    frame = f['X'][0]
    label = f['Y'][0]
    snr = f['Z'][0]

# === Decodifica della modulazione ===
modulation_names = [
    'OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32',
    'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64',
    'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC',
    'AM_DSB_SC', 'FM', 'GMSK', 'OQPS'
]
mod_name = modulation_names[np.argmax(label)]

# === Ricostruzione del segnale reale ===
I = frame[:, 0]
Q = frame[:, 1]
t = np.arange(len(I)) / fs  # asse temporale

# Ricostruzione: I cos + Q sin
signal_reconstructed = I * np.cos(2 * np.pi * fc * t) + Q * np.sin(2 * np.pi * fc * t)

# === Plot ===
plt.figure(figsize=(12, 4))
plt.plot(t * 1e3, signal_reconstructed)
plt.title(f"Segnale reale ricostruito - {mod_name} @ {snr} dB")
plt.xlabel("Tempo [ms]")
plt.ylabel("Ampiezza")
plt.grid(True)
plt.tight_layout()
plt.show()
