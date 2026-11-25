"""Pure-Python ONNX Protobuf (IR v8 / Opset 17) serializer and PyTorch->ONNX exporter.

Allows exporting trained PyTorch CNNs (including depthwise-separable MobileNetV3 blocks
and YOLOv8-style grid detectors) directly into valid FP32 and INT8-quantized `.onnx` files
loadable by `onnxruntime.InferenceSession` without requiring the external `onnx` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn


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


def _fold_conv_bn(conv: nn.Conv2d, bn: Optional[nn.BatchNorm2d]) -> Tuple[np.ndarray, np.ndarray]:
    w = conv.weight.detach().cpu().numpy().astype(np.float32)
    if conv.bias is not None:
        b = conv.bias.detach().cpu().numpy().astype(np.float32)
    else:
        b = np.zeros(conv.out_channels, dtype=np.float32)

    if bn is None:
        return w, b

    gamma = bn.weight.detach().cpu().numpy().astype(np.float32)
    beta = bn.bias.detach().cpu().numpy().astype(np.float32)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    var = bn.running_var.detach().cpu().numpy().astype(np.float32)
    eps = float(bn.eps)

    inv_std = gamma / np.sqrt(var + eps)
    w_folded = w * inv_std.reshape(-1, 1, 1, 1)
    b_folded = (b - mean) * inv_std + beta
    return w_folded, b_folded


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


def export_sequential_cnn_to_onnx(
    module: nn.Module,
    input_shape: Tuple[int, int, int, int],
    output_shape: Sequence[int | str],
    output_path: Path,
    graph_name: str = "visionguard_net",
    quantize_int8: bool = False,
) -> Path:
    """Export a PyTorch module (composed of Conv2d/BatchNorm2d/ReLU/MaxPool2d/AdaptiveAvgPool2d/Flatten/Linear)
    into a standard ONNX file with optional INT8 weight quantization."""
    module.eval()
    builder = OnnxGraphBuilder.create(graph_name)
    builder.add_input("input", list(input_shape), elem_type=1)

    layers = list(module.modules())[1:]  # skip root container, walk flat children cleanly
    children: List[nn.Module] = []

    def _collect(m: nn.Module) -> None:
        if len(list(m.children())) == 0:
            children.append(m)
        else:
            for c in m.children():
                _collect(c)

    _collect(module)
    cur = "input"
    idx = 0
    while idx < len(children):
        layer = children[idx]
        if isinstance(layer, nn.Conv2d):
            bn = None
            if idx + 1 < len(children) and isinstance(children[idx + 1], nn.BatchNorm2d):
                bn = children[idx + 1]
                idx += 1
            w, b = _fold_conv_bn(layer, bn)
            stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            pad = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
            cur = builder.add_conv(
                cur,
                w,
                b,
                stride=stride,
                padding=pad,
                groups=int(layer.groups),
                quantize_int8=quantize_int8,
            )
        elif isinstance(layer, nn.ReLU):
            cur = builder.add_relu(cur)
        elif isinstance(layer, nn.Sigmoid):
            cur = builder.add_sigmoid(cur)
        elif isinstance(layer, nn.MaxPool2d):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            s = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            p = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
            cur = builder.add_maxpool(cur, kernel_size=k, stride=s, padding=p)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            cur = builder.add_global_avg_pool(cur)
        elif isinstance(layer, nn.Flatten):
            cur = builder.add_flatten(cur, axis=int(layer.start_dim))
        elif isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = (
                layer.bias.detach().cpu().numpy().astype(np.float32)
                if layer.bias is not None
                else np.zeros(layer.out_features, dtype=np.float32)
            )
            cur = builder.add_linear(cur, w, b, quantize_int8=quantize_int8)
        elif isinstance(layer, (nn.Dropout, nn.Identity)):
            pass
        else:
            raise ValueError(f"Unsupported layer type for ONNX export: {type(layer).__name__}")
        idx += 1

    builder.add_identity(cur, "output")
    builder.add_output("output", list(output_shape), elem_type=1)
    model_bytes = builder.build_model_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_bytes)
    return output_path
