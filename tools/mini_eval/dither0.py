# 强制所有 WavFrontend 实例在调用时 dither=0（generate 路径用的是另一个实例）
from funasr.frontends.wav_frontend import WavFrontend
_orig = WavFrontend.__call__
def _call(self, *a, **k):
    self.dither = 0.0
    return _orig(self, *a, **k)
WavFrontend.__call__ = _call
