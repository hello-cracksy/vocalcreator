from . import _utils

class VowelGenerator:
    def __init__(self, formants, sr, cutoff=4500):
        self.formants = formants
        self.sr = sr
        self.cutoff = cutoff

    # ===== 共通ユーティリティ =====
    def _pulse(self, freq, t):
        return _utils.bandlimit_pulse(freq, t, cutoff=self.cutoff)

    def _slice(self, wave, start, end=None):
        s = int(self.sr * start)
        e = int(self.sr * end) if end else None
        return wave[s:e]

    def _f(self, wave, key):
        if key not in self.formants:
            raise ValueError(f"Unknown formant key: {key}")
        return _utils.bpf(wave, self.formants[key])

    def _norm(self, x):
        return _utils.normalize(x)

    # ===== 基本母音 =====
    def generate(self, freq, t, vowel):
        base = self._pulse(freq, t)
        return self._norm(self._f(base, vowel))

    # ===== u系 =====
    def u_generate(self, freq, t, vowel):
        base = self._pulse(freq, t)
        return self._norm(
            _utils.crossfade_add(
                self._f(self._slice(base, 0, 0.06), "u"),
                self._f(self._slice(base, 0.06), vowel)
            )
        )

    def nu_generate(self, freq, t, vowel):
        base = self._pulse(freq, t)

        n = (
            self._f(self._slice(base, 0, 0.05), "u") +
            self._f(self._slice(base, 0, 0.05), "e")
        )
        u = self._f(self._slice(base, 0.05, 0.08), "u")
        v = self._f(self._slice(base, 0.08), vowel)

        return self._norm(_utils.crossfade_add_many([n, u, v]))

    # ===== ny =====
    def ny_generate(self, freq, t, vowel):
        base = self._pulse(freq, t)

        n = self._f(self._slice(base, 0, 0.04), "n")
        y = self._f(self._slice(base, 0.04, 0.08), "y")
        u = self._f(self._slice(base, 0.08, 0.11), "u")
        v = self._f(self._slice(base, 0.11), vowel)

        return self._norm(_utils.crossfade_add_many([n, y, u, v]))

    # ===== y =====
    def y_generate(self, freq, t, vowel):
        base = self._pulse(freq, t)

        y = self._f(self._slice(base, 0, 0.06), "y")
        u = self._f(self._slice(base, 0.06, 0.12), "u")
        v = self._f(self._slice(base, 0.12), vowel)

        return self._norm(_utils.crossfade_add_many([y, u, v]))

    # ===== w =====
    def w_generate(self, freq, t, vowel):
        base = self._pulse(freq, t)

        w = self._f(self._slice(base, 0, 0.06), "w")
        u = self._f(self._slice(base, 0.06, 0.12), "u")
        v = self._f(self._slice(base, 0.12), vowel)

        return self._norm(_utils.crossfade_add_many([w, u, v]))
