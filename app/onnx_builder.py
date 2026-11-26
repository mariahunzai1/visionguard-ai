"""Pure-Python ONNX Protobuf (IR v8 / Opset 17) graph serializer.

Provides `OnnxGraphBuilder` for constructing valid FP32 and INT8-quantized `.onnx`
graphs loadable by `onnxruntime.InferenceSession` without requiring `onnx` or `torch`
at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np


def _varint(n: int) -> bytes:
    if n < 0:
        n &= (1 << 64) - 1
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _field_varint(num: int, val: int) -> bytes:
    return _varint((num << 3) | 0) + _varint(val)


def _field_bytes(num: int, payload: bytes) -> bytes:
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _field_str(num: int, text: str) -> bytes:
    return _field_bytes(num, text.encode("utf-8"))


def _make_tensor_proto(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    dtype_map = {
        np.float32: 1,
        np.uint8: 2,
        np.int8: 3,
        np.int32: 6,
        np.int64: 7,
    }
    if arr.dtype.type not in dtype_map:
        raise TypeError(f"Unsupported tensor dtype: {arr.dtype}")
    dt = dtype_map[arr.dtype.type]
    buf = bytearray()
    for d in arr.shape:
        buf += _field_varint(1, int(d))
    buf += _field_varint(2, dt)
    buf += _field_str(8, name)
    buf += _field_bytes(9, arr.tobytes())
    return bytes(buf)


def _make_value_info(name: str, elem_type: int, dims: Sequence[int | str]) -> bytes:
    shape_buf = bytearray()
    for d in dims:
        if isinstance(d, int):
            dim_msg = _field_varint(1, d)
        else:
            dim_msg = _field_str(2, str(d))
        shape_buf += _field_bytes(1, dim_msg)
    tensor_type = _field_varint(1, elem_type) + _field_bytes(2, bytes(shape_buf))
    type_proto = _field_bytes(1, tensor_type)
    return _field_str(1, name) + _field_bytes(2, type_proto)


def _attr_int(name: str, value: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT=2)
    return _field_str(1, name) + _field_varint(3, value) + _field_varint(20, 2)


def _attr_ints(name: str, values: Sequence[int]) -> bytes:
    # AttributeProto: name=1, ints=8 (packed or repeated), type=20 (INTS=7)
    buf = bytearray()
    buf += _field_str(1, name)
    for v in values:
        buf += _field_varint(8, int(v))
    buf += _field_varint(20, 7)
    return bytes(buf)


def _attr_float(name: str, value: float) -> bytes:
    import struct

    # AttributeProto: name=1, f=4 (wire type 5 fixed32), type=20 (FLOAT=1)
    return (
        _field_str(1, name)
        + _varint((4 << 3) | 5)
        + struct.pack("<f", float(value))
        + _field_varint(20, 1)
    )


def _make_node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    attrs: Sequence[bytes] = (),
    name: str = "",
) -> bytes:
    buf = bytearray()
    for inp in inputs:
        buf += _field_str(1, inp)
    for out in outputs:
        buf += _field_str(2, out)
    if name:
        buf += _field_str(3, name)
    buf += _field_str(4, op_type)
    for attr in attrs:
        buf += _field_bytes(5, attr)
    return bytes(buf)


@dataclass
class OnnxGraphBuilder:
    graph_name: str
    nodes: List[bytes]
    initializers: List[bytes]
    inputs: List[bytes]
    outputs: List[bytes]
    counter: int = 0

    @classmethod
    def create(cls, graph_name: str) -> "OnnxGraphBuilder":
        return cls(graph_name=graph_name, nodes=[], initializers=[], inputs=[], outputs=[])

    def _next_name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def add_input(self, name: str, shape: Sequence[int | str], elem_type: int = 1) -> None:
        self.inputs.append(_make_value_info(name, elem_type, shape))

    def add_output(self, name: str, shape: Sequence[int | str], elem_type: int = 1) -> None:
        self.outputs.append(_make_value_info(name, elem_type, shape))

    def add_initializer(self, name: str, arr: np.ndarray) -> None:
        self.initializers.append(_make_tensor_proto(name, arr))

    def add_conv(
        self,
        x_name: str,
        weight: np.ndarray,
        bias: np.ndarray,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
        quantize_int8: bool = False,
    ) -> str:
        w_name = self._next_name("conv_w")
        b_name = self._next_name("conv_b")
        y_name = self._next_name("conv_out")

        if quantize_int8:
            max_abs = float(np.max(np.abs(weight)))
            scale = np.float32(max(max_abs / 127.0, 1e-7))
            w_q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
            zp = np.int8(0)
            s_name = f"{w_name}_scale"
            zp_name = f"{w_name}_zp"
            w_dq_name = f"{w_name}_dq"
            self.add_initializer(w_name, w_q)
            self.add_initializer(s_name, np.array(scale, dtype=np.float32))
            self.add_initializer(zp_name, np.array(zp, dtype=np.int8))
            self.nodes.append(
                _make_node("DequantizeLinear", [w_name, s_name, zp_name], [w_dq_name])
            )
            weight_input = w_dq_name
        else:
            self.add_initializer(w_name, weight.astype(np.float32))
            weight_input = w_name

        self.add_initializer(b_name, bias.astype(np.float32))
        kh, kw = weight.shape[2], weight.shape[3]
        ph, pw = padding
        attrs = [
            _attr_ints("kernel_shape", [kh, kw]),
            _attr_ints("strides", [stride[0], stride[1]]),
            _attr_ints("pads", [ph, pw, ph, pw]),
            _attr_int("group", groups),
        ]
        self.nodes.append(_make_node("Conv", [x_name, weight_input, b_name], [y_name], attrs))
        return y_name

    def add_relu(self, x_name: str) -> str:
        y_name = self._next_name("relu")
        self.nodes.append(_make_node("Relu", [x_name], [y_name]))
        return y_name

    def add_sigmoid(self, x_name: str) -> str:
        y_name = self._next_name("sigmoid")
        self.nodes.append(_make_node("Sigmoid", [x_name], [y_name]))
        return y_name

    def add_maxpool(
        self,
        x_name: str,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
    ) -> str:
        y_name = self._next_name("maxpool")
        attrs = [
            _attr_ints("kernel_shape", [kernel_size, kernel_size]),
            _attr_ints("strides", [stride, stride]),
            _attr_ints("pads", [padding, padding, padding, padding]),
        ]
        self.nodes.append(_make_node("MaxPool", [x_name], [y_name], attrs))
        return y_name

    def add_global_avg_pool(self, x_name: str) -> str:
        y_name = self._next_name("gap")
        self.nodes.append(_make_node("GlobalAveragePool", [x_name], [y_name]))
        return y_name

    def add_flatten(self, x_name: str, axis: int = 1) -> str:
        y_name = self._next_name("flatten")
        self.nodes.append(_make_node("Flatten", [x_name], [y_name], [_attr_int("axis", axis)]))
        return y_name

    def add_linear(
        self,
        x_name: str,
        weight: np.ndarray,
        bias: np.ndarray,
        quantize_int8: bool = False,
    ) -> str:
        w_name = self._next_name("fc_w")
        b_name = self._next_name("fc_b")
        y_name = self._next_name("gemm")

        if quantize_int8:
            max_abs = float(np.max(np.abs(weight)))
            scale = np.float32(max(max_abs / 127.0, 1e-7))
            w_q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
            zp = np.int8(0)
            s_name = f"{w_name}_scale"
            zp_name = f"{w_name}_zp"
            w_dq_name = f"{w_name}_dq"
            self.add_initializer(w_name, w_q)
            self.add_initializer(s_name, np.array(scale, dtype=np.float32))
            self.add_initializer(zp_name, np.array(zp, dtype=np.int8))
            self.nodes.append(
                _make_node("DequantizeLinear", [w_name, s_name, zp_name], [w_dq_name])
            )
            weight_input = w_dq_name
        else:
            self.add_initializer(w_name, weight.astype(np.float32))
            weight_input = w_name

        self.add_initializer(b_name, bias.astype(np.float32))
        attrs = [
            _attr_float("alpha", 1.0),
            _attr_float("beta", 1.0),
            _attr_int("transB", 1),
        ]
        self.nodes.append(_make_node("Gemm", [x_name, weight_input, b_name], [y_name], attrs))
        return y_name

    def add_identity(self, x_name: str, out_name: str) -> str:
        self.nodes.append(_make_node("Identity", [x_name], [out_name]))
        return out_name

    def build_model_bytes(self) -> bytes:
        graph = bytearray()
        for n in self.nodes:
            graph += _field_bytes(1, n)
        graph += _field_str(2, self.graph_name)
        for init in self.initializers:
            graph += _field_bytes(5, init)
        for inp in self.inputs:
            graph += _field_bytes(11, inp)
        for out in self.outputs:
            graph += _field_bytes(12, out)

        opset = _field_str(1, "") + _field_varint(2, 17)
        model = bytearray()
        model += _field_varint(1, 8)  # ir_version = 8
        model += _field_str(2, "VisionGuard-ONNX-Exporter")
        model += _field_str(3, "1.0")
        model += _field_bytes(7, bytes(graph))
        model += _field_bytes(8, opset)
        return bytes(model)
