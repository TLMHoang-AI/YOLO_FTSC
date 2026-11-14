"""Offline TensorRT API contract tests for local Ultralytics fork.

This is deliberately a fake execution-context test (no CUDA engine needed).
Run on a Python installation with torch+ultralytics fork:
    python -m unittest utils.tests.test_ftsc_trt_execution_api_v1
"""
from __future__ import annotations

import unittest
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from unittest.mock import patch


class FakeNamedContext:
    def __init__(self, bind_ok=True, exec_ok=True):
        self.bind_ok = bind_ok
        self.exec_ok = exec_ok
        self.tensor_addresses = OrderedDict()
        self.stream = None
        self.legacy_used = False

    def set_tensor_address(self, name, ptr):
        self.tensor_addresses[name] = ptr
        return self.bind_ok

    def execute_async_v3(self, stream_handle):
        self.stream = stream_handle
        return self.exec_ok

    def execute_v2(self, bindings):
        self.legacy_used = True
        raise AssertionError("Legacy binding-list execute_v2 is forbidden for TensorRT 10/11")


class FakeLegacyContext:
    def __init__(self):
        self.bound = None
        self.named_used = False

    def execute_v2(self, bindings):
        self.bound = bindings
        return True

    def set_tensor_address(self, *args):
        self.named_used = True
        raise AssertionError("Unexpected named I/O for TensorRT 7/8/9")


class TensorRTExecutionAPITests(unittest.TestCase):
    def build_backend(self, named=True, context=None):
        import torch
        from ultralytics.nn.backends.tensorrt import TensorRTBackend

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        fake = TensorRTBackend.__new__(TensorRTBackend)
        fake.is_trt10 = named
        fake.dynamic = False
        fake.device = torch.device("cpu")
        inp = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        out = torch.zeros((1, 5, 16), dtype=torch.float32)
        fake.bindings = OrderedDict((
            ("images", Binding("images", inp.numpy().dtype, tuple(inp.shape), inp, inp.data_ptr())),
            ("output0", Binding("output0", out.numpy().dtype, tuple(out.shape), out, out.data_ptr())),
        ))
        fake.output_names = ["output0"]
        fake.binding_addrs = OrderedDict((k, b.ptr) for k, b in fake.bindings.items())
        fake.context = context or (FakeNamedContext() if named else FakeLegacyContext())
        return fake, inp, out

    def test_tensor_rt_10_uses_named_io_and_v3(self):
        import torch
        model, input_tensor, out = self.build_backend()
        with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=123)):
            actual = model.forward(input_tensor)
        self.assertIs(actual[0], out)
        self.assertFalse(model.context.legacy_used)
        self.assertEqual(model.context.stream, 123)
        self.assertEqual(model.context.tensor_addresses["images"], input_tensor.data_ptr())
        self.assertEqual(model.context.tensor_addresses["output0"], out.data_ptr())

    def test_tensor_rt_10_fails_closed_when_execution_fails(self):
        import torch
        model, inp, _ = self.build_backend(context=FakeNamedContext(exec_ok=False))
        with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=123)):
            with self.assertRaisesRegex(RuntimeError, "execute_async_v3"):
                model.forward(inp)

    def test_tensor_rt_7_to_9_keeps_legacy_execution(self):
        model, inp, out = self.build_backend(named=False)
        result = model.forward(inp)
        self.assertIs(result[0], out)
        self.assertEqual(model.context.bound, [inp.data_ptr(), out.data_ptr()])
        self.assertFalse(model.context.named_used)


if __name__ == "__main__":
    unittest.main()
