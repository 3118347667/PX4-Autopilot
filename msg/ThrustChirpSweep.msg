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
