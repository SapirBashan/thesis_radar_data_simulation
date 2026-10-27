"""Self-contained, versioned byte-stream codecs for floating or ADC I/Q blocks."""
from dataclasses import dataclass
import json
import math
import struct
import zlib

import numpy as np
import zstandard as zstd

MAGIC = b"RADARC01"
MAX_BYTES = 512 * 1024 * 1024


@dataclass
class SampleBlock:
    data: np.ndarray
    adc_step: float | None = None

    def __post_init__(self):
        self.data = np.ascontiguousarray(self.data)
        if self.data.dtype == np.dtype('<c8') and self.data.ndim == 3:
            if self.adc_step is not None:
                raise ValueError('Floating reference has no ADC scale')
        elif self.data.dtype == np.dtype('<i2') and self.data.ndim == 4 and self.data.shape[-1] == 2:
            if self.adc_step is None or not np.isfinite(self.adc_step) or self.adc_step <= 0:
                raise ValueError('ADC reference needs a positive finite step')
        else:
            raise ValueError('Use little-endian complex64 [E,R,P] or int16 [E,R,P,IQ]')
        if not self.data.size or self.data.nbytes > MAX_BYTES:
            raise ValueError('Empty or oversized sample block')

    @property
    def count(self):
        return math.prod(self.data.shape[:3])

    def iq(self):
        if self.adc_step is None:
            return self.data
        return ((self.data[..., 0].astype(np.float64) + 1j * self.data[..., 1]) * self.adc_step).astype(np.complex64)


def _pack(header, payload):
    header = dict(header, payload_bytes=len(payload), crc32=zlib.crc32(payload))
    meta = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    return MAGIC + struct.pack('<I', len(meta)) + meta + payload


def inspect(blob):
    if len(blob) < 12 or blob[:8] != MAGIC:
        raise ValueError('Invalid stream magic/version')
    length = struct.unpack('<I', blob[8:12])[0]
    if not 0 < length <= 1024 * 1024 or 12 + length > len(blob):
        raise ValueError('Invalid metadata length')
    header = json.loads(blob[12:12 + length])
    payload = blob[12 + length:]
    if len(payload) != header['payload_bytes'] or zlib.crc32(payload) != header['crc32']:
        raise ValueError('Truncated or corrupted payload')
    shape = header['shape']
    if not isinstance(shape, list) or len(shape) not in (3, 4) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Invalid shape')
    if header['dtype'] not in ('<c8', '<i2'):
        raise ValueError('Unsupported dtype')
    expected = math.prod(shape) * np.dtype(header['dtype']).itemsize
    if expected > MAX_BYTES:
        raise ValueError('Oversized decoded array')
    return header, payload, expected


def _symbols_to_bytes(symbols, bits):
    bit_matrix = np.unpackbits(symbols.astype('<u2').view(np.uint8).reshape(-1, 2), axis=1, bitorder='little')
    return np.packbits(bit_matrix[:, :bits].reshape(-1), bitorder='little').tobytes()


def _bytes_to_symbols(payload, bits, count):
    if len(payload) != (count * bits + 7) // 8:
        raise ValueError('Invalid packed symbol length')
    raw = np.unpackbits(np.frombuffer(payload, np.uint8), bitorder='little')
    if np.any(raw[count * bits:]):
        raise ValueError('Nonzero padding bits')
    return (raw[:count * bits].reshape(count, bits).astype(np.uint16) * (1 << np.arange(bits, dtype=np.uint16))).sum(axis=1).astype(np.int32)


def decode(blob):
    """Decode solely from the stored bytes; no encoder instance/state is needed."""
    h, payload, expected = inspect(blob)
    shape, dtype, method = h['shape'], np.dtype(h['dtype']), h['method']
    if method in ('zstd', 'predictive_zstd'):
        if zstd.frame_content_size(payload) != expected:
            raise ValueError('Unexpected zstd output size')
        payload = zstd.ZstdDecompressor().decompress(payload, max_output_size=expected, allow_extra_data=False)
    if method in ('raw', 'zstd', 'predictive_zstd'):
        if len(payload) != expected:
            raise ValueError('Invalid raw byte count')
        data = np.frombuffer(payload, dtype=dtype).copy().reshape(shape)
        if method == 'predictive_zstd':
            if dtype == np.dtype('<c8'):
                words = data.view('<u4').reshape(*shape, 2)
                data = np.bitwise_xor.accumulate(words, axis=2).view('<c8').reshape(shape)
            else:
                words = data.view('<u2')
                data = np.add.accumulate(words, axis=2, dtype=np.uint16).view('<i2')
    elif method == 'baq':
        bits, block_size = h['bits'], h['block_complex_samples']
        if type(bits) is not int or not 2 <= bits <= 12 or type(block_size) is not int or block_size < 1:
            raise ValueError('Invalid BAQ settings')
        count = math.prod(shape[:3]) * 2
        nblocks = (count + 2 * block_size - 1) // (2 * block_size)
        if len(payload) < nblocks * 4:
            raise ValueError('Missing block scales')
        scales = np.frombuffer(payload[:4 * nblocks], '<f4')
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError('Invalid BAQ scales')
        limit = (1 << (bits - 1)) - 1
        codes = _bytes_to_symbols(payload[4 * nblocks:], bits, count)
        if np.any(codes > 2 * limit):
            raise ValueError('Reserved BAQ code')
        scalars = (codes - limit) * np.repeat(scales, 2 * block_size)[:count].astype(np.float64)
        if dtype == np.dtype('<c8'):
            data = scalars.astype('<f4').view('<c8').reshape(shape)
        else:
            data = np.clip(np.rint(scalars), -32768, 32767).astype('<i2').reshape(shape)
    else:
        raise ValueError('Unknown codec')
    return SampleBlock(data, h['adc_step'])


@dataclass(frozen=True)
class Codec:
    method: str
    level: int = 3
    bits: int = 4
    block_complex_samples: int = 128

    def __post_init__(self):
        if self.method not in ('raw', 'zstd', 'predictive_zstd', 'baq'):
            raise ValueError('Unknown codec')
        if not 2 <= self.bits <= 12 or self.block_complex_samples < 1 or not 1 <= self.level <= 22:
            raise ValueError('Invalid codec settings')

    @property
    def name(self):
        if self.method == 'baq':
            return f'baq_b{self.bits}_n{self.block_complex_samples}'
        return self.method + (f'_l{self.level}' if self.method != 'raw' else '')

    @property
    def lossless(self):
        return self.method != 'baq'

    def encode(self, block):
        data = block.data
        header = dict(method=self.method, dtype=data.dtype.str, shape=list(data.shape), adc_step=block.adc_step)
        if self.method == 'baq':
            size = 2 * self.block_complex_samples
            scalars = data.view('<f4').reshape(-1).astype(np.float64) if block.adc_step is None else data.reshape(-1).astype(np.float64)
            if not np.all(np.isfinite(scalars)):
                raise ValueError('BAQ requires finite samples')
            limit = (1 << (self.bits - 1)) - 1
            starts = np.arange(0, len(scalars), size)
            peaks = np.maximum.reduceat(np.abs(scalars), starts)
            scales = (peaks / limit).astype('<f4')
            scales[peaks == 0] = 1
            if np.any(scales <= 0):
                raise ValueError('BAQ scale underflow')
            quantized = np.clip(np.rint(scalars / np.repeat(scales, size)[:len(scalars)]), -limit, limit).astype(np.int32)
            payload = scales.tobytes() + _symbols_to_bytes(quantized + limit, self.bits)
            header.update(bits=self.bits, block_complex_samples=self.block_complex_samples)
        else:
            if self.method == 'predictive_zstd':
                # Predict along slow time. Bitwise XOR on IEEE-754 words is
                # reversible even for signed zero/NaN; arithmetic float deltas aren't.
                if block.adc_step is None:
                    words = data.view('<u4').reshape(*data.shape, 2)
                    residual = words.copy()
                    residual[:, :, 1:] = words[:, :, 1:] ^ words[:, :, :-1]
                else:
                    words = data.view('<u2')
                    residual = words.copy()
                    residual[:, :, 1:] = words[:, :, 1:] - words[:, :, :-1]
                payload = residual.tobytes()
            else:
                payload = data.tobytes()
            if self.method != 'raw':
                header['level'] = self.level
                payload = zstd.ZstdCompressor(level=self.level, write_checksum=True).compress(payload)
        return _pack(header, payload)

    def decode(self, blob):
        return decode(blob)
