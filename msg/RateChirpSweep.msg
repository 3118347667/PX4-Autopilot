uint64 timestamp        # time since system start (microseconds)
uint64 timestamp_sample # timestamp of the data sample on which this message is based (microseconds)

float32 chirp           # chirp signal added to the normalized torque setpoint
float32 r               # rate controller reference
float32 u               # controller output with chirp signal
float32 y               # angular acceleration response

uint64 esc_timestamp    # timestamp of the latest ESC status sample (microseconds)
uint8 ESC_RPM_MAX = 8
uint8 esc_count         # number of connected ESCs reported by esc_status
uint8 esc_online_flags  # bitmask indicating which ESC is online
int32[8] esc_rpm        # ESC motor RPM, indexed as esc_status.esc[]
