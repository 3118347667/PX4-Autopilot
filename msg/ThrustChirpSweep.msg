uint64 timestamp         # time since system start (microseconds)
uint64 timestamp_sample  # timestamp of the data sample on which this message is based (microseconds)

# Z thrust chirp sweep response for frequency domain system identification
# r: body Z acceleration setpoint (including gravity)
# u: thrust setpoint with chirp
# y: measured body Z acceleration (including gravity)

float32 chirp            # chirp signal (normalized thrust)
float32 r                # body Z acceleration setpoint (m/s^2)
float32 u                # thrust setpoint with chirp (normalized, body z)
float32 y                # measured body Z acceleration (m/s^2)

uint64 esc_timestamp     # timestamp of the latest ESC status sample (microseconds)
uint8 ESC_RPM_MAX = 8
uint8 esc_count          # number of connected ESCs reported by esc_status
uint8 esc_online_flags   # bitmask indicating which ESC is online
int32[8] esc_rpm         # ESC motor RPM, indexed as esc_status.esc[]
