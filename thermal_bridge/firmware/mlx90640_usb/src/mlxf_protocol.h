// MLXF v1 wire format (MCU -> host), shared contract with
// thermal_bridge/mlx90640_protocol.py. All multi-byte fields little-endian.
//
//  off  size  field
//   0    4    magic "MLXF"
//   4    1    version = 1
//   5    1    sensor_id (0 = narrow / Wire, 1 = wide / Wire1)
//   6    2    seq (u16, per sensor, wraps)
//   8    4    mcu_millis (u32)
//  12    4    ta (f32, sensor ambient degC)
//  16    2    n = 768
//  18  2*n    int16 temperatures, centi-degC, row-major 24 x 32 (sensor order)
//  18+2n 2    CRC16-CCITT (poly 0x1021, init 0xFFFF, no reflection, no xorout)
//             over bytes [4, 18+2n) i.e. everything after the magic.
//
// Text lines (responses / status) start with '#' and end with '\n'; they are
// always written between packets, never inside one.
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

#define MLXF_VERSION 1
#define MLXF_NPIX 768
#define MLXF_HEADER_LEN 18
#define MLXF_PACKET_LEN (MLXF_HEADER_LEN + 2 * MLXF_NPIX + 2)  // 1556

static inline uint16_t mlxf_crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc = 0xFFFF) {
  while (len--) {
    crc ^= (uint16_t)(*data++) << 8;
    for (uint8_t b = 0; b < 8; b++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

static inline void mlxf_put_u16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
static inline void mlxf_put_u32(uint8_t *p, uint32_t v) {
  p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF; p[2] = (v >> 16) & 0xFF; p[3] = v >> 24;
}

static inline int16_t mlxf_centi(float t) {
  if (isnan(t)) return INT16_MIN;  // INT16_MIN = invalid pixel marker
  float c = t * 100.0f;
  if (c > 32767.0f) return 32767;
  if (c < -32767.0f) return -32767;
  return (int16_t)lroundf(c);
}

// Encodes one packet into out[MLXF_PACKET_LEN]. Returns the packet length.
static inline size_t mlxf_encode(uint8_t *out, uint8_t sensor_id, uint16_t seq, uint32_t millis_now,
                                 float ta, const float *temps) {
  memcpy(out, "MLXF", 4);
  out[4] = MLXF_VERSION;
  out[5] = sensor_id;
  mlxf_put_u16(out + 6, seq);
  mlxf_put_u32(out + 8, millis_now);
  uint32_t ta_bits;
  memcpy(&ta_bits, &ta, 4);  // RP2040 / ESP32 are little-endian IEEE-754
  mlxf_put_u32(out + 12, ta_bits);
  mlxf_put_u16(out + 16, MLXF_NPIX);
  uint8_t *p = out + MLXF_HEADER_LEN;
  for (int i = 0; i < MLXF_NPIX; i++, p += 2) mlxf_put_u16(p, (uint16_t)mlxf_centi(temps[i]));
  mlxf_put_u16(p, mlxf_crc16_ccitt(out + 4, (size_t)(p - (out + 4))));
  return MLXF_PACKET_LEN;
}
