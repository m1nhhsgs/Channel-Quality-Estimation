# Packet Format

Firmware owner: Hiep

The current project does not validate payload content. Packet loss is measured by missing
sequence numbers.

```cpp
typedef struct __attribute__((packed)) {
    char name[16];
    uint32_t seq;
    uint32_t txTimeMs;
    float value;
} DataPacket;
```

Serial output from RX:

```csv
RX,session_id,rx_ms_receiver,src_mac,dst_mac,seq,tx_ms_sender,gap,total_missing,name,value,rssi_dbm,noise_floor_dbm,snr_db,channel,sig_mode,mcs,rate
```

Example:

```csv
RX,run01,10234,F0:24:F9:0E:47:C4,F0:24:F9:0E:9C:F4,1250,9811,0,0,ESP_SENDER,12.30,-67,-95,28,1,0,0,0
```

Default runtime configuration:

```text
send_interval_ms = 50
packet_rate      = 20 packets/second
payload_size     = 28 bytes
channel          = 1
```
