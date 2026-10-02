// mlx90640_usb -- two Melexis MLX90640 (narrow BAA 55x35 deg, wide BAB 110x75 deg)
// on two I2C buses -> USB CDC serial, binary "MLXF" v1 packets (see mlxf_protocol.h).
//
// Sensor 0 (narrow) = Wire, sensor 1 (wide) = Wire1, both at the default address 0x33.
// Each sensor is served by its own execution context so they run at full rate:
//   RP2040  : core0 = sensor 0 + command parser, core1 = sensor 1 (arduino-pico setup1/loop1)
//   ESP32-S3: one FreeRTOS task per sensor, loop() = command parser
//
// Text commands (one per line, '\n' or '\r\n'), responses are '#'-prefixed JSON lines:
//   RATE <hz>    MLX refresh rate (subpage rate): 0.5 1 2 4 8 16 32 64. Full frame = 2 subpages.
//   EMIS <e>     emissivity 0.10 .. 1.00 (default 0.95)
//   INFO?        board / config / per-sensor status
// A missing or failing sensor never stops the firmware: it is reported in INFO and
// re-probed every 2 s; no packets are sent for it meanwhile.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

#include "mlxf_protocol.h"

#if defined(MLX_BOARD_PICO)
#include <pico/mutex.h>
#define FW_BOARD "pico"
static const int PIN_SDA0 = 4, PIN_SCL0 = 5;  // I2C0 -> Wire  -> narrow
static const int PIN_SDA1 = 6, PIN_SCL1 = 7;  // I2C1 -> Wire1 -> wide
static mutex_t tx_mutex;
static void tx_lock_init() { mutex_init(&tx_mutex); }
static void tx_lock() { mutex_enter_blocking(&tx_mutex); }
static void tx_unlock() { mutex_exit(&tx_mutex); }
#elif defined(MLX_BOARD_ESP32S3)
#define FW_BOARD "esp32s3"
static const int PIN_SDA0 = 8, PIN_SCL0 = 9;    // Wire  -> narrow
static const int PIN_SDA1 = 17, PIN_SCL1 = 18;  // Wire1 -> wide
static SemaphoreHandle_t tx_mutex;
static void tx_lock_init() { tx_mutex = xSemaphoreCreateMutex(); }
static void tx_lock() { xSemaphoreTake(tx_mutex, portMAX_DELAY); }
static void tx_unlock() { xSemaphoreGive(tx_mutex); }
#else
#error "define MLX_BOARD_PICO or MLX_BOARD_ESP32S3 (see platformio.ini)"
#endif

#define FW_VERSION "1.0.0"
static const uint32_t I2C_HZ_EEPROM = 400000;  // EEPROM dump in begin(): conservative
static const uint32_t I2C_HZ_RUN = 1000000;    // FM+: needed to read 832 words fast enough at 16 Hz
static const float DEFAULT_EMISSIVITY = 0.95f;
static const float LIB_EMISSIVITY = 0.95f;     // hard-coded in Adafruit_MLX90640::getFrame()
static const uint32_t REPROBE_MS = 2000;
static const uint8_t MAX_CONSECUTIVE_ERRORS = 5;

enum SensorState : uint8_t { ST_MISSING = 0, ST_OK, ST_ERROR };
static const char *state_name(SensorState s) {
  return s == ST_OK ? "ok" : (s == ST_ERROR ? "error" : "missing");
}

struct Sensor {
  uint8_t id;
  const char *name;
  TwoWire *wire;
  Adafruit_MLX90640 mlx;
  volatile SensorState state;
  volatile uint32_t frames;
  volatile uint32_t errors;
  volatile float ta;
  uint16_t seq;
  uint8_t consecutive_errors;
  uint32_t last_probe_ms;
  uint8_t applied_rate;  // mlx90640_refreshrate_t, 0xFF = not yet applied
  float frame[MLXF_NPIX];
  uint8_t packet[MLXF_PACKET_LEN];
};

static Sensor sensors[2] = {
    {0, "narrow", &Wire},
    {1, "wide", &Wire1},
};

// Shared config, written by the command parser, read by the sensor contexts.
static volatile uint8_t cfg_rate = MLX90640_8_HZ;
static volatile float cfg_emissivity = DEFAULT_EMISSIVITY;

static const float RATE_HZ[] = {0.5f, 1, 2, 4, 8, 16, 32, 64};
static volatile bool boot_done = false;  // RP2040: setup1/loop1 start in parallel with setup()

// ---------------------------------------------------------------- output
static void send_text_line(const char *line) {
  tx_lock();
  Serial.print('#');
  Serial.print(line);
  Serial.print('\n');
  tx_unlock();
}

static void send_packet(const uint8_t *buf, size_t len) {
  tx_lock();
  Serial.write(buf, len);
  tx_unlock();
}

// ---------------------------------------------------------------- sensor
static bool probe(Sensor &s) {
  s.wire->beginTransmission(MLX90640_I2CADDR_DEFAULT);
  return s.wire->endTransmission() == 0;
}

static bool sensor_init(Sensor &s) {
  // Probe first: Adafruit_MLX90640::begin() allocates an I2C device object,
  // so only call it when a sensor actually answers (no leak while unplugged).
  if (!probe(s)) return false;
  s.wire->setClock(I2C_HZ_EEPROM);
  bool ok = s.mlx.begin(MLX90640_I2CADDR_DEFAULT, s.wire);
  s.wire->setClock(I2C_HZ_RUN);
  if (!ok) return false;
  s.mlx.setMode(MLX90640_CHESS);  // factory calibration mode, best accuracy
  s.mlx.setResolution(MLX90640_ADC_18BIT);
  s.mlx.setRefreshRate((mlx90640_refreshrate_t)cfg_rate);
  s.applied_rate = cfg_rate;
  s.consecutive_errors = 0;
  return true;
}

// The library computes To with emissivity 0.95 and Tr = Ta - 8 (open air).
// Re-map to the configured emissivity using the Melexis To equation:
//   To^4 = I/(e*A) + taTr(e),  taTr(e) = Tr^4 - (Tr^4 - Ta^4)/e
// => To^4 = (e0/e) * (To0^4 - taTr(e0)) + taTr(e). (ksTo range term ignored, < 0.1 K.)
static void apply_emissivity(float *t, float ta, float e) {
  if (fabsf(e - LIB_EMISSIVITY) < 1e-4f) return;
  const float K = 273.15f;
  float ta4 = powf(ta + K, 4), tr4 = powf(ta - OPENAIR_TA_SHIFT + K, 4);
  float tatr0 = tr4 - (tr4 - ta4) / LIB_EMISSIVITY;
  float tatr = tr4 - (tr4 - ta4) / e;
  float scale = LIB_EMISSIVITY / e;
  for (int i = 0; i < MLXF_NPIX; i++) {
    float x = scale * (powf(t[i] + K, 4) - tatr0) + tatr;
    t[i] = x > 0 ? sqrtf(sqrtf(x)) - K : NAN;
  }
}

static void sensor_fail(Sensor &s) {
  s.errors++;
  if (++s.consecutive_errors >= MAX_CONSECUTIVE_ERRORS) {
    s.state = ST_MISSING;  // unplugged / dead bus: fall back to periodic re-probe
    s.last_probe_ms = millis();
  } else {
    s.state = ST_ERROR;
  }
}

// One full frame (both chess subpages) for one sensor; blocks ~2/rate s.
static void sensor_step(Sensor &s) {
  if (s.state == ST_MISSING) {
    if (millis() - s.last_probe_ms < REPROBE_MS) {
      delay(10);
      return;
    }
    s.last_probe_ms = millis();
    if (!sensor_init(s)) return;
    s.state = ST_OK;
  }
  if (s.applied_rate != cfg_rate) {
    s.mlx.setRefreshRate((mlx90640_refreshrate_t)cfg_rate);
    s.applied_rate = cfg_rate;
  }
  if (s.mlx.getFrame(s.frame) != 0) {
    sensor_fail(s);
    return;
  }
  float ta = s.mlx.getTa(false);  // ambient from the frame just read
  if (isnan(ta) || ta < -45.0f || ta > 130.0f) {  // garbage read (bus glitch)
    sensor_fail(s);
    return;
  }
  apply_emissivity(s.frame, ta, cfg_emissivity);
  s.ta = ta;
  s.state = ST_OK;
  s.consecutive_errors = 0;
  s.frames++;
  size_t n = mlxf_encode(s.packet, s.id, s.seq++, millis(), ta, s.frame);
  send_packet(s.packet, n);
}

// ---------------------------------------------------------------- commands
static void send_info() {
  char buf[640];
  int n = snprintf(buf, sizeof(buf),
                   "{\"fw\":\"mlx90640_usb\",\"version\":\"" FW_VERSION "\",\"proto\":%d,\"board\":\"" FW_BOARD
                   "\",\"rate_hz\":%.1f,\"emissivity\":%.3f,\"mode\":\"chess\",\"i2c_hz\":%lu,\"sensors\":[",
                   MLXF_VERSION, RATE_HZ[cfg_rate], (double)cfg_emissivity, (unsigned long)I2C_HZ_RUN);
  for (int i = 0; i < 2 && n < (int)sizeof(buf); i++) {
    Sensor &s = sensors[i];
    n += snprintf(buf + n, sizeof(buf) - n,
                  "%s{\"id\":%d,\"name\":\"%s\",\"state\":\"%s\",\"serial\":\"%04X%04X%04X\",\"frames\":%lu,"
                  "\"errors\":%lu,\"ta\":%.2f}",
                  i ? "," : "", s.id, s.name, state_name(s.state), s.mlx.serialNumber[0], s.mlx.serialNumber[1],
                  s.mlx.serialNumber[2], (unsigned long)s.frames, (unsigned long)s.errors, (double)s.ta);
  }
  if (n < (int)sizeof(buf)) snprintf(buf + n, sizeof(buf) - n, "]}");
  send_text_line(buf);
}

static void reply(bool ok, const char *cmd, const char *detail) {
  char buf[160];
  snprintf(buf, sizeof(buf), "{\"ok\":%s,\"cmd\":\"%s\",\"detail\":\"%s\"}", ok ? "true" : "false", cmd, detail);
  send_text_line(buf);
}

static void handle_command(char *line) {
  char *arg = strchr(line, ' ');
  if (arg) *arg++ = '\0';
  if (strcmp(line, "INFO?") == 0) {
    send_info();
  } else if (strcmp(line, "RATE") == 0 && arg) {
    float hz = atof(arg);
    for (uint8_t i = 0; i < sizeof(RATE_HZ) / sizeof(RATE_HZ[0]); i++) {
      if (fabsf(RATE_HZ[i] - hz) < 0.01f) {
        cfg_rate = i;
        reply(true, "RATE", arg);
        return;
      }
    }
    reply(false, "RATE", "allowed: 0.5 1 2 4 8 16 32 64");
  } else if (strcmp(line, "EMIS") == 0 && arg) {
    float e = atof(arg);
    if (e >= 0.1f && e <= 1.0f) {
      cfg_emissivity = e;
      reply(true, "EMIS", arg);
    } else {
      reply(false, "EMIS", "range 0.1 .. 1.0");
    }
  } else {
    reply(false, line, "unknown command");
  }
}

static void poll_commands() {
  static char line[64];
  static uint8_t len = 0;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[len] = '\0';
      if (len) handle_command(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    } else {
      len = 0;  // overlong line: drop it
    }
  }
}

// ---------------------------------------------------------------- setup / loop
static void sensor_boot(Sensor &s) {
  s.state = ST_MISSING;
  s.applied_rate = 0xFF;
  s.ta = NAN;
  s.last_probe_ms = millis();
  if (sensor_init(s)) s.state = ST_OK;
}

void setup() {
  tx_lock_init();
  Serial.begin(115200);  // USB CDC: baud rate is ignored
#if defined(MLX_BOARD_ESP32S3)
  Serial.setTxTimeoutMs(0);  // never block when the host is not reading
#endif
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 1500) delay(10);

#if defined(MLX_BOARD_PICO)
  Wire.setSDA(PIN_SDA0);
  Wire.setSCL(PIN_SCL0);
  Wire1.setSDA(PIN_SDA1);
  Wire1.setSCL(PIN_SCL1);
  Wire.begin();
  Wire1.begin();
  Wire.setClock(I2C_HZ_RUN);
  Wire1.setClock(I2C_HZ_RUN);
#else
  Wire.begin(PIN_SDA0, PIN_SCL0, I2C_HZ_RUN);
  Wire1.begin(PIN_SDA1, PIN_SCL1, I2C_HZ_RUN);
#endif
  for (auto &s : sensors) sensor_boot(s);
  send_info();
  boot_done = true;

#if defined(MLX_BOARD_ESP32S3)
  for (int i = 0; i < 2; i++) {
    xTaskCreatePinnedToCore(
        [](void *arg) {
          Sensor *s = (Sensor *)arg;
          for (;;) {
            sensor_step(*s);
            vTaskDelay(1);  // let the idle task / watchdog breathe
          }
        },
        sensors[i].name, 8192, &sensors[i], 1, nullptr, i);
  }
#endif
}

void loop() {
  poll_commands();
#if defined(MLX_BOARD_PICO)
  sensor_step(sensors[0]);
#else
  delay(5);
#endif
}

#if defined(MLX_BOARD_PICO)
void setup1() {
  while (!boot_done) delay(1);
}
void loop1() { sensor_step(sensors[1]); }
#endif
