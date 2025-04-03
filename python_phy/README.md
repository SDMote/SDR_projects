# BLE & IEEE 802.15.4 Modulation/Demodulation in Python

This folder contains Python scripts for demodulating BLE and IEEE 802.15.4 packets from IQ baseband measurements, and modulating from the contents of the physical payload.

It also contains an automated process to estimate the amplitude, phase and time-shift of an assume known interference within a measured IQ packet, and subtract it to recover the affected packet. 

## Prerequisites

GNU Radio must be installed to run the demodulators, which use the [Symbol Sync implementation](https://wiki.gnuradio.org/index.php/Symbol_Sync).
The code has been tested with GNU Radio v3.11.0.0git-843-g6b25c171 (Python 3.10.12).
For installation instructions, see: [GNU Radio Wiki](https://wiki.gnuradio.org/index.php/InstallingGR).
