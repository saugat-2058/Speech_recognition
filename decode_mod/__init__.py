import os
from .decode_cffi import ffi as _ffi
import logging


def open_dll():
    dlldir = os.path.abspath(os.path.dirname(__file__))
    os.environ["PATH"] = dlldir + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dlldir)
    return _ffi.dlopen(os.path.join(dlldir, "libvosk.dll"))

_c = open_dll()


class Model:
    def __init__(self, model_path=None, model_name=None, lang=None):
        if model_path is not None:
            self._handle = _c.vosk_model_new(model_path.encode("utf-8"))
        else:
            model_path = self.get_model_path(model_name, lang)
            self._handle = _c.vosk_model_new(model_path.encode("utf-8"))
        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a model")

    def get_model_path(self, model_name, lang):
        if model_name is None:
            model_path = self.get_model_by_lang(lang)
        else:
            model_path = self.get_model_by_name(model_name)
        return str(model_path)
    def get_handle(self):
        return self._handle

class DecodeModel:
    def __init__(self, *args):
        if len(args) == 2:
            self._handle = _c.vosk_recognizer_new(args[0]._handle, args[1])
        else:
            raise TypeError("Unknown arguments")

        if self._handle == _ffi.NULL:
            raise Exception("Failed to create a recognizer")

    def __del__(self):
        _c.vosk_recognizer_free(self._handle)

    def AcceptWaveform(self, data):
        res = _c.vosk_recognizer_accept_waveform(self._handle, data, len(data))
        if res < 0:
            raise Exception("Failed to process waveform")
        return res

    def Result(self):
        return _ffi.string(_c.vosk_recognizer_result(self._handle)).decode("utf-8")

    def FinalResult(self):
        return _ffi.string(_c.vosk_recognizer_final_result(self._handle)).decode("utf-8")

def SetLogLevel(level):
    return _c.vosk_set_log_level(level)


def GpuInit():
    _c.vosk_gpu_init()

def GpuThreadInit():
    _c.vosk_gpu_thread_init()
