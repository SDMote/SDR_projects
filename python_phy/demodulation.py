from gnuradio import digital, blocks, gr
import numpy as np


# Computes instantaneous frequency of a complex IQ signal
def quadrature_demod(iq_samples: np.ndarray, gain: float | int = 1) -> np.ndarray:
    return np.diff(np.unwrap(np.angle(iq_samples))) * gain


# Hard decision slicsdf
def binary_slicer(data: np.ndarray) -> np.ndarray:
    return np.where(data >= 0, 1, 0).astype(np.int8)


# Symbol synchronisation function from GNU Radio with the following Timing Error Detectors
# TED_MUELLER_AND_MULLER
# TED_MOD_MUELLER_AND_MULLER
# TED_ZERO_CROSSING
# TED_GARDNER
# TED_EARLY_LATE
# TED_DANDREA_AND_MENGALI_GEN_MSK
# TED_MENGALI_AND_DANDREA_GMSK
# TED_SIGNAL_TIMES_SLOPE_ML
# TED_SIGNUM_TIMES_SLOPE_ML
def symbol_sync(
    input_samples: np.ndarray,
    sps: float,
    TED_gain: float = 1.0,
    loop_BW: float = 0.045,
    damping: float = 1.0,
    max_deviation: float = 1.5,
    out_sps: int = 1,
) -> np.ndarray:
    # Convert NumPy array to GNU Radio format
    src = blocks.vector_source_f(input_samples.tolist(), False, 1, [])

    # Instantiate the symbol sync block
    symbol_sync_block = digital.symbol_sync_ff(
        digital.TED_MOD_MUELLER_AND_MULLER,
        sps,
        loop_BW,
        damping,
        TED_gain,
        max_deviation,
        out_sps,
        digital.constellation_bpsk().base(),
        digital.IR_MMSE_8TAP,
        128,
        [],
    )

    sink = blocks.vector_sink_f(1, 1024 * 4)  # Sink to collect the output

    # Connect blocks and run
    tb = gr.top_block()
    tb.connect(src, symbol_sync_block)
    tb.connect(symbol_sync_block, sink)
    tb.run()

    return np.array(sink.data())
