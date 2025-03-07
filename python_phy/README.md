# BLE & IEEE 802.15.4 Demodulation in Python

This folder contains Python scripts for demodulating BLE and IEEE 802.15.4 packets from IQ baseband measurements.

## Prerequisites

GNU Radio must be installed to run the demodulators, which use the [Symbol Sync implementation](https://wiki.gnuradio.org/index.php/Symbol_Sync).
The code has been tested with GNU Radio v3.11.0.0git-843-g6b25c171 (Python 3.10.12).
For installation instructions, see: [GNU Radio Wiki](https://wiki.gnuradio.org/index.php/InstallingGR).

## TODO

- Implement controlled interference subtraction (in progress; currently working with a tone).
- Support intermediate frequency (non-baseband) demodulation.
- Integrate matched filtering in the demodulation process for improved performance at low SNR.
- Evaluate demodulation performance under low SNR conditions.
