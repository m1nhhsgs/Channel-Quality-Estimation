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

DataPacket txData;
uint32_t seqSend = 0;

// Sender MAC (tham khảo): F0:24:F9:0E:47:C4
// Receiver MAC (tham khảo): F0:24:F9:0E:9C:F4

// MAC cua board RECEIVER
uint8_t peerMac[] = {0xF0, 0x24, 0xF9, 0x0E, 0x9C, 0xF4};

const char* deviceName = "ESP_SENDER";
const char* sessionId = "run01";
const uint8_t espNowChannel = 1;
const uint32_t sendIntervalMs = 50;  // 20 packets/second
uint32_t lastSendMs = 0;

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

void onSent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  Serial.print("TX_CB,");
  Serial.print(sessionId);
  Serial.print(",");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");

  if (tx_info) {
    Serial.print(",");
    Serial.print(tx_info->tx_status == WIFI_SEND_SUCCESS ? "OK" : "FAIL");
    Serial.print(",");
    Serial.print((int)tx_info->ifidx);
    Serial.print(",");
    Serial.print((int)tx_info->rate);
    Serial.print(",");
    Serial.print((int)tx_info->data_len);
    Serial.print(",");
    printMac(tx_info->des_addr);
    Serial.print(",");
    printMac(tx_info->src_addr);
  } else {
    Serial.print(",NA,NA,NA,NA,NA,NA");
  }

  Serial.println();
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

  esp_now_register_send_cb(onSent);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, peerMac, 6);
  peerInfo.channel = espNowChannel;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Add peer failed");
    return;
  }

  Serial.print("Role: SENDER | Peer MAC: ");
  printMac(peerMac);
  Serial.println();
  Serial.print("Session: ");
  Serial.println(sessionId);
  Serial.print("Send interval ms: ");
  Serial.println(sendIntervalMs);
  Serial.println("CSV header:");
  Serial.println("record,session_id,ms,seq,tx_ms,to,name,value,payload_len,send_err");
  Serial.println("record,session_id,ms,status,drv_status,ifidx,rate,payload_len,to,from");
  Serial.println("ESP-NOW ready");
}

void loop() {
  uint32_t now = millis();
  if (now - lastSendMs < sendIntervalMs) {
    delay(1);
    return;
  }
  lastSendMs = now;

  memset(&txData, 0, sizeof(txData));
  strncpy(txData.name, deviceName, sizeof(txData.name) - 1);
  txData.seq = ++seqSend;
  txData.txTimeMs = now;
  txData.value = random(0, 1000) / 10.0;

  esp_err_t sendErr = esp_now_send(peerMac, (uint8_t*)&txData, sizeof(txData));

  Serial.print("TX_REQ,");
  Serial.print(sessionId);
  Serial.print(",");
  Serial.print(now);
  Serial.print(",");
  Serial.print(txData.seq);
  Serial.print(",");
  Serial.print(txData.txTimeMs);
  Serial.print(",");
  printMac(peerMac);
  Serial.printf(",%s,%.2f,%u,%d\n",
                txData.name,
                txData.value,
                (unsigned)sizeof(txData),
                (int)sendErr);
}
