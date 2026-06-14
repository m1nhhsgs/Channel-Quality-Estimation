#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_mac.h>

typedef struct __attribute__((packed)) {
  char name[16];
  uint32_t seq;
  uint32_t txTimeMs;
  float value;
} DataPacket;

const char* deviceName = "ESP_RECEIVER";
const char* sessionId = "run01";
const uint8_t espNowChannel = 1;
bool hasLastSeq = false;
uint32_t lastSeq = 0;
uint32_t totalMissing = 0;

bool isZeroMac(const uint8_t *mac) {
  if (!mac) return true;
  for (int i = 0; i < 6; ++i) {
    if (mac[i] != 0) return false;
  }
  return true;
}

bool getStaMac(uint8_t outMac[6]) {
  if (!outMac) return false;

  if (esp_wifi_get_mac(WIFI_IF_STA, outMac) == ESP_OK && !isZeroMac(outMac)) {
    return true;
  }

  if (esp_read_mac(outMac, ESP_MAC_WIFI_STA) == ESP_OK && !isZeroMac(outMac)) {
    return true;
  }

  return false;
}

void printMac(const uint8_t *mac) {
  if (!mac) {
    Serial.print("??:??:??:??:??:??");
    return;
  }

  Serial.printf("%02X:%02X:%02X:%02X:%02X:%02X",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void onRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (!info) return;
  if (len != (int)sizeof(DataPacket)) {
    Serial.printf("RX_DROP,%s,%lu,len_mismatch,%d,%u\n",
                  sessionId, (unsigned long)millis(), len, (unsigned)sizeof(DataPacket));
    return;
  }

  DataPacket rx;
  memcpy(&rx, data, sizeof(rx));

  uint32_t rxMs = millis();
  uint32_t gap = 0;

  if (hasLastSeq && rx.seq > lastSeq + 1) {
    gap = rx.seq - lastSeq - 1;
    totalMissing += gap;
  }
  if (!hasLastSeq || rx.seq > lastSeq) {
    lastSeq = rx.seq;
    hasLastSeq = true;
  }

  int rssi = 0;
  int noiseFloor = 0;
  int snr = 0;
  unsigned channel = 0;
  unsigned sigMode = 0;
  unsigned mcs = 0;
  unsigned rate = 0;

  if (info->rx_ctrl) {
    const wifi_pkt_rx_ctrl_t *r = info->rx_ctrl;
    rssi = (int)r->rssi;
    noiseFloor = (int)r->noise_floor;
    snr = rssi - noiseFloor;
    channel = (unsigned)r->channel;
    sigMode = (unsigned)r->sig_mode;
    mcs = (unsigned)r->mcs;
    rate = (unsigned)r->rate;
  }

  Serial.print("RX,");
  Serial.print(sessionId);
  Serial.print(",");
  Serial.print(rxMs);
  Serial.print(",");
  printMac(info->src_addr);
  Serial.print(",");
  printMac(info->des_addr);
  Serial.printf(",%lu,%lu,%lu,%lu,%s,%.2f,%d,%d,%d,%u,%u,%u,%u\n",
                (unsigned long)rx.seq,
                (unsigned long)rx.txTimeMs,
                (unsigned long)gap,
                (unsigned long)totalMissing,
                rx.name,
                rx.value,
                rssi,
                noiseFloor,
                snr,
                channel,
                sigMode,
                mcs,
                rate);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  esp_err_t chErr = esp_wifi_set_channel(espNowChannel, WIFI_SECOND_CHAN_NONE);
  Serial.printf("WiFi channel set: %u | err=%d\n", (unsigned)espNowChannel, (int)chErr);

  uint8_t myMac[6] = {0};
  Serial.print("My MAC: ");
  if (getStaMac(myMac)) {
    printMac(myMac);
    Serial.println();
  } else {
    Serial.println("00:00:00:00:00:00");
    Serial.println("Warning: cannot read valid STA MAC");
  }

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  esp_now_register_recv_cb(onRecv);

  Serial.print("Role: RECEIVER | Name: ");
  Serial.println(deviceName);
  Serial.print("Session: ");
  Serial.println(sessionId);
  Serial.println("CSV header:");
  Serial.println("record,session_id,rx_ms_receiver,src_mac,dst_mac,seq,tx_ms_sender,gap,total_missing,name,value,rssi_dbm,noise_floor_dbm,snr_db,channel,sig_mode,mcs,rate");
  Serial.println("record,session_id,ms,reason,got_len,expected_len");
  Serial.println("ESP-NOW ready");
}

void loop() {
  delay(1000);
}
