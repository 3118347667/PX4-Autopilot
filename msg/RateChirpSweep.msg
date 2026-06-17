uint64 timestamp        # time since system start (microseconds)
uint64 timestamp_sample # timestamp of the data sample on which this message is based (microseconds)

float32 chirp           # chirp signal added to the normalized torque setpoint
float32 r               # rate controller reference
float32 u               # controller output with chirp signal
float32 y               # angular acceleration response
