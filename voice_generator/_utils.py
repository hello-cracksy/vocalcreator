from functools import lru_cache
import numpy as np
from scipy.signal import square, butter, sosfilt, stft, istft, find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def plot_wave(signal,sr):
    N = len(signal)
    t = np.linspace(0,N/sr,N)
    plt.plot(t,signal)
    plt.show()


def plot_fft(signal, sr, lim=10000):
    """
    signal : 1次元配列（波形）
    sr     : サンプリングレート [Hz]
    """
    N = len(signal)

    # FFT
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(N, d=1/sr)

    # 振幅スペクトル
    amp = np.abs(fft) / N

    # 描画
    plt.figure()
    plt.plot(freq, amp)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim(0,lim)
    plt.grid(True)
    plt.show()


def detect_formants(signal, sr):
    """フォルマントの推定用"""
    f, t, Z = stft(signal, sr, nperseg=2048)
    spec = np.mean(np.abs(Z), axis=1)

    # 平滑化（重要）
    spec_smooth = gaussian_filter1d(spec, sigma=5)

    # ピーク検出
    peaks, _ = find_peaks(
        spec_smooth,
        distance=5,      # 近すぎるピークを除外
        prominence=0.001   # 目立つ山だけ残す
    )

    return f[peaks], spec_smooth, f

def normalize(wave, volume=1):
    m = np.max(np.abs(wave))
    if m > 0:
        wave = wave / m * volume
    return wave.astype(np.float32)

@lru_cache(maxsize=None)
def _get_sos(fc, bw, SR=SR):
    low = max(fc - bw / 2, 1)       # 0Hz以下にならないように
    high = min(fc + bw / 2, SR/2-1) # ナイキスト以下に制限
    return butter(2, [low, high], btype='band', fs=SR, output='sos')


def bpf(wave, formant):
    """キャッシュ版BPF"""
    wave = wave.astype(np.float32)
    filtered = np.zeros_like(wave, dtype=np.float32)
    for fc, bw in formant:
        sos = _get_sos(fc, bw)
        filtered += sosfilt(sos, wave)
    # 正規化
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered = filtered / max_val

    return filtered


def _apply_fade(wave, fade_time, SR=44100):
    """
    wave       : np.ndarray 波形
    fade_time  : float フェード時間[秒]
    SR         : int   サンプリングレート[Hz]

    戻り値
    wave       : フェードをかけた波形
    fade_samples : フェードに使用したサンプル数
    """
    wave = np.atleast_1d(wave).astype(np.float32)
    fade_samples = int(SR * fade_time)

    # 波形が短い場合はフェードを短縮
    fade_samples = min(fade_samples, len(wave)//2)

    if fade_samples > 0:
        fade_in  = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out

    return wave, fade_samples


def crossfade_add(noise, wave, fade_time=0.05, SR=44100):
    """
    noise : np.ndarray ノイズ波形
    wave  : np.ndarray 合成波形
    fade_time : float フェード時間[秒]
    SR    : サンプリングレート

    戻り値
    wave_buffer : np.ndarray フェード重ね合わせ済み波形
    """
    # フェード適用
    noise, fade_n = _apply_fade(noise, fade_time, SR)
    wave, fade_w  = _apply_fade(wave, fade_time, SR)

    # フェード重なりは短い方を使う
    overlap = min(fade_n, fade_w)

    # 出力バッファ長
    length = len(noise) + len(wave) - overlap
    wave_buffer = np.zeros(length, dtype=np.float32)

    # ノイズをセット
    wave_buffer[:len(noise)] += noise

    # 波形を重ねる
    start_idx = len(noise) - overlap
    wave_buffer[start_idx : start_idx + len(wave)] += wave

    return wave_buffer


def crossfade_add_many(waves, fade_time=0.05, SR=44100):
    """
    waves : [np.ndarray, np.ndarray, ...]  連結したい波形配列
    fade_time : float
    SR : int
    戻り値 : np.ndarray
    """
    if len(waves) == 0:
        return np.array([], dtype=np.float32)
    if len(waves) == 1:
        return waves[0].astype(np.float32)

    out = waves[0].astype(np.float32)

    for i in range(1, len(waves)):
        next_wave = waves[i].astype(np.float32)

        # フェード
        a, fa = _apply_fade(out, fade_time, SR)
        b, fb = _apply_fade(next_wave, fade_time, SR)

        overlap = min(fa, fb)

        length = len(a) + len(b) - overlap
        buf = np.zeros(length, dtype=np.float32)

        buf[:len(a)] += a
        start = len(a) - overlap
        buf[start:start + len(b)] += b

        out = buf

    return out


def bandlimit_pulse(freq, t, cutoff=5000, SR=SR):
    """矩形波＋軽いローパスフィルタ"""
    wave = square(2 * np.pi * freq * t, duty=0.125).astype(np.float32)
    sos = butter(16, cutoff, btype='low', fs=SR, output='sos')  # 次数を34→16に下げた
    return sosfilt(sos, wave)
