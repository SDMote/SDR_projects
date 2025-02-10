%% Recovery of IEEE 802.15.4 OQPSK Signals
% https://mathworks.com/help/comm/ug/recovery-of-ieee-802-15-4-oqpsk-signals.html

%% Open data
fid = fopen('capture_nRF/data/802154_short_includeCRC.dat', 'rb'); % Open file for reading in binary mode
raw_data = fread(fid, 'float32'); % Read as 32-bit float
fclose(fid);

% Convert interleaved real and imaginary parts into complex values
data = raw_data(1:2:end) + 1j * raw_data(2:2:end);

% Slice data to contain only packet
data = data(3200:6000);

% Plot IQ samples
plot(real(data), 'DisplayName', 'I');
hold on;
plot(imag(data), 'DisplayName', 'Q');
hold off;
legend;
title('Time Domain Signal');
xlabel('Sample Index');
ylabel('Amplitude');
grid on;

%% Matched filtering (half-sine shaping)
spc = 10; % samples per symbol (sampled at 10MHz, 1 QPSK Msymbol/s)
decimationFactor = 1; %
half_sine_pulse = sin(0:pi/spc:pi);
fprintf('half_sine_pulse = %d samples\n', ...
    length(half_sine_pulse));

matched_filter = dsp.FIRDecimator(decimationFactor,half_sine_pulse);
% len(halfSinePulse) = sps + 1. Apparently doesn't have to be precise
% for demodulation purposees, only for modulation (see GNU Radio modulation implementation)
% filteredOQPSK = matched_filter(data); % matched filter output
filteredOQPSK = data;

% % Plot IQ samples
% plot(real(filteredOQPSK), 'DisplayName', 'I');
% hold on;
% plot(imag(filteredOQPSK), 'DisplayName', 'Q');
% plot(real(data), 'DisplayName', 'Idata');
% plot(imag(data), 'DisplayName', 'Idata');
% plot((half_sine_pulse), 'DisplayName', 'halfSinePulse');
% hold off;
% legend;
% title('Time Domain Signal');
% xlabel('Sample Index');
% ylabel('Amplitude');
% grid on;

%% Delaying back Q component (or delaying I to be causal)
% Frequency offsets up to 30 kHz are expected from IEEE 802.15.4 PHY receivers
% To observe the constellation of OQPSK signals, delay back the Q component half a symbol

% Plot constellation of QPSK-equivalent (impaired)
% received signal
function qpskSig = helperAlignIQ(oqpskSig, sps)

qpskSig = complex(real(oqpskSig(1:end-sps/2)), ...
    imag(oqpskSig(sps/2+1:end)));
end
% PROBLEM: THIS ASSUMES THAT THE SPS IS EVEN!
% PROBLEM: implementing a "backwards" delay is non-causal.
% I would have to instead delay the I component half a symbol, which is equivalent

filteredQPSK_delayed = helperAlignIQ(filteredOQPSK, spc/decimationFactor);

% Plot IQ samples
plot(real(filteredQPSK_delayed), 'DisplayName', 'I');
hold on;
plot(imag(filteredQPSK_delayed), 'DisplayName', 'Q delayed');
plot(real(filteredOQPSK), 'DisplayName', 'I');
plot(imag(filteredOQPSK), 'DisplayName', 'Q');

hold off;
legend;
title('Time Domain Signal');
xlabel('Sample Index');
ylabel('Amplitude');
grid on;

% This function does some black magic. Somehow it manages to allign the symbols, even though the phase is unknown
constellation = comm.ConstellationDiagram( ...
    'XLimits',[-7.5 7.5], ...
    'YLimits',[-7.5 7.5], ...
    'ReferenceConstellation',5*qammod(0:3, 4), ...
    'Name','Received QPSK-Equivalent Signal');
constellation.Position = [constellation.Position(1:2) 300 300];
constellation(filteredQPSK);

%% plot squared FFT
% Square the signal and compute its FFT
Xfft = fftshift(fft(filteredQPSK_delayed.^2));

% Plot FFT magnitude
figure;
plot(abs(Xfft));
xlabel('Frequency Bins');
ylabel('Magnitude');
title('FFT Magnitude of Squared Signal');
grid on;



%% Coarse frequency compensation (replacable by FLL-Band edge)
% This is another method that can't be implemented in real time.
% It basically inteprets the data  as MSK, and finds the two MSK frequency peaks.
% Then it averages the values
% This could be replaced with a FLL-Band edge, which kind of does the same but automatically (remember balance)
% Coarse frequency compensation of OQPSK signal

%For OQPSK modulation, the coarse frequency compensator uses the FFT-based algorithm described in [ 4 ].
% The algorithm searches for spectral peaks at ±200 kHz around the symbol rate.


coarseFrequencyCompensator = comm.CoarseFrequencyCompensator( ...
    'Modulation','OQPSK', ...
    'SampleRate',spc*1e6/decimationFactor, ...
    'FrequencyResolution',1e3);
[coarseCompensatedOQPSK, coarseFrequencyOffset] = ...
    coarseFrequencyCompensator(filteredOQPSK);

% print estimated offset
fprintf('Estimated frequency offset = %.3f kHz\n', ...
    coarseFrequencyOffset/1000);

% Plot QPSK-equivalent coarsely compensated signal
% Again, black magic made with black magic boxes in matlab, but basically a IQ plot with a phase correction
coarseCompensatedQPSK = helperAlignIQ(coarseCompensatedOQPSK, spc/decimationFactor);
release(constellation);
constellation.Name = ...
    'Coarse frequency compensation (QPSK-Equivalent)';
constellation(coarseCompensatedQPSK);

%% Fine frequency compensation (This is Costas Loop in GNU Radio, or also included in Costellation Receiver block)
% This is done, however, after Symbol Sync in the GNU Radio QPSK diagrams
% According to the MATLAB tutorial, this algorithm is different than the one used for QPSK, which
% does not apply to OQPSK signals even if their I component is delayed by half a symbol
% Reference Book: Rice, Michael. Digital Communications - A Discrete-Time Approach. 1st ed. New York, NY: Prentice Hall, 2008.

% Fine frequency compensation of OQPSK signal
fineFrequencyCompensator = comm.CarrierSynchronizer( ...
    'Modulation','OQPSK', ...
    'SamplesPerSymbol',spc/decimationFactor);
fineCompensatedOQPSK = ...
    fineFrequencyCompensator(coarseCompensatedOQPSK);

% Plot QPSK-equivalent finely compensated signal
fineCompensatedQPSK = helperAlignIQ(fineCompensatedOQPSK, spc/decimationFactor);
release(constellation);
constellation.Name = 'Fine frequency compensation (QPSK-Equivalent)';
constellation(fineCompensatedQPSK);


%% Timing recovery (Symbol sync in GNU Radio)
% This IS equivalent to the timing recovery counterpart for QPSK signals (so, just Symbol Sync)
% Timing recovery of OQPSK signal, via its QPSK-equivalent version
symbolSynchronizer = comm.SymbolSynchronizer( ...
    'Modulation','OQPSK', ...
    'SamplesPerSymbol',spc/decimationFactor);
syncedQPSK = symbolSynchronizer(fineCompensatedOQPSK);

% Plot QPSK symbols (1 sample per chip)
release(constellation);
constellation.Name = 'Timing Recovery (QPSK-Equivalent)';
constellation(syncedQPSK);
