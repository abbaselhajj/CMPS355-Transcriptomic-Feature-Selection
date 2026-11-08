"""Read the *specific* XDR R serialization used by the supplied ibd_datasets.RData.

This is deliberately a minimal, strict RDX3/XDR decoder, not a general R parser.
It supports ordinary integer/numeric/character vectors, factors, data.frames,
attributes, pairlists, symbols and shared symbol references. No R code is executed.
Unsupported types raise an error. For general RData use R's load() or rdata.
"""
from __future__ import annotations
import gzip
import struct
from pathlib import Path
import numpy as np

class RSerializationError(ValueError):
    pass

class _XDR:
    def __init__(self, compressed_path: Path):
        self.bytes = gzip.open(compressed_path, "rb").read()
        self.buf = memoryview(self.bytes)
        self.pos = 0
        self.references: list[object] = []

    def take(self, n: int) -> memoryview:
        if n < 0 or self.pos + n > len(self.buf):
            raise RSerializationError(f"Invalid R serialization length {n} at {self.pos}")
        result = self.buf[self.pos:self.pos+n]
        self.pos += n
        return result

    def int32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def value(self, depth: int = 0):
        if depth > 100:
            raise RSerializationError("Unexpectedly deep R object graph")
        info = self.int32()
        kind = info & 255
        has_attributes = bool(info & 512)
        has_tag = bool(info & 1024)
        if kind in (0, 250, 251, 252, 253, 254):
            return None
        if kind == 255:  # REFSXP: 1-based symbol reference
            index = info >> 8 or self.int32()
            if not 1 <= index <= len(self.references):
                raise RSerializationError(f"Invalid shared reference {index}")
            return self.references[index - 1]
        if kind == 1:  # SYMSXP
            index = len(self.references)
            self.references.append(None)
            name = self.value(depth + 1)
            symbol = {"SYM": name}
            self.references[index] = symbol
            return symbol
        if kind == 9:  # CHARSXP
            length = self.int32()
            return None if length < 0 else bytes(self.take(length)).decode("utf-8")
        if kind in (10, 13, 14, 15, 24):
            length = self.int32()
            datatype = {10: ">i4", 13: ">i4", 14: ">f8", 15: ">c16", 24: "u1"}[kind]
            array = np.frombuffer(self.take(length * np.dtype(datatype).itemsize), dtype=datatype)
            data = array
        elif kind in (16, 19, 20):  # character, generic and expression vectors
            length = self.int32()
            if length < 0:
                raise RSerializationError("Negative vector length")
            data = [self.value(depth+1) for _ in range(length)]
        elif kind in (2, 6, 17):  # pairlists
            attr = self.value(depth+1) if has_attributes else None
            tag = self.value(depth+1) if has_tag else None
            first = self.value(depth+1)
            rest = self.value(depth+1)
            return {"pair": first, "next": rest, "tag": tag, "attrs": attr}
        else:
            raise RSerializationError(f"Unsupported R object type {kind} at byte {self.pos-4}")
        attrs = self.value(depth+1) if has_attributes else None
        if has_attributes:
            return {"data": data, "attrs": attrs, "object": bool(info & 256)}
        return data

    def load(self) -> dict:
        if bytes(self.take(5)) != b"RDX3\n" or bytes(self.take(2)) != b"X\n":
            raise RSerializationError("Expected compressed RDX3 XDR RData")
        version, writer, min_reader = self.int32(), self.int32(), self.int32()
        enc_len = self.int32()
        encoding = bytes(self.take(enc_len)).decode("ascii")
        if version != 3 or encoding != "UTF-8":
            raise RSerializationError(f"Unsupported R serialization version or encoding: {version}/{encoding}")
        parsed = self.value()
        if self.pos != len(self.buf):
            raise RSerializationError("Trailing unread RData bytes")
        return pairlist_dict(parsed)

def pairlist_dict(node: object) -> dict:
    out = {}
    while isinstance(node, dict) and "pair" in node:
        tag = node["tag"]
        if not isinstance(tag, dict) or "SYM" not in tag:
            raise RSerializationError("Expected tagged R pairlist")
        key = tag["SYM"]
        if key in out:
            raise RSerializationError(f"Duplicated R pairlist key: {key}")
        out[key] = node["pair"]
        node = node["next"]
    if node is not None:
        raise RSerializationError("Malformed R pairlist tail")
    return out

def read_ibd_rdata(path: str | Path) -> dict:
    obj = _XDR(Path(path)).load()
    expected = {f"{prefix}_d{i}" for prefix in ("count", "desc") for i in range(1, 5)}
    if set(obj) != expected:
        raise RSerializationError(f"Unexpected RData object names {sorted(set(obj) ^ expected)}")
    return obj
