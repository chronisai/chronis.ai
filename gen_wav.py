import wave, struct, math
sample_rate = 24000
duration = 3
frequency = 220
samples = [int(32767 * math.sin(2 * math.pi * frequency * t / sample_rate)) for t in range(sample_rate * duration)]
with wave.open('test_voice.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
print('Created test_voice.wav')