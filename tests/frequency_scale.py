from megfractal.utils import scale2freq, fband2scale

sfreq = 100

j1 = 3
j2 = 6

fband_input = (1, 10)

fband_exact = (scale2freq(j2, sfreq), scale2freq(j1, sfreq))

j1_out, j2_out = fband2scale(fband_input, sfreq)

assert (j1, j2) == (j1_out, j2_out)

j1_out, j2_out = fband2scale(fband_exact, sfreq)

assert (j1_out, j2_out) == (j1, j2)
