
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except Exception:
    ON_PI = False

class GPIOController:
    def __init__(self, cfg):
        self.enabled = cfg['gpio'].get('enabled', True)
        self.pin = int(cfg['gpio'].get('pin', 17))
        self.active_high = bool(cfg['gpio'].get('active_high', True))
        if ON_PI and self.enabled:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
    def trigger(self, pulse_s=1.0):
        if not self.enabled:
            print('GPIO disabled in config - skipping trigger')
            return
        if not ON_PI:
            print(f'[SIM] GPIO trigger pin {self.pin} for {pulse_s}s')
            return
        try:
            val = GPIO.HIGH if self.active_high else GPIO.LOW
            GPIO.output(self.pin, val)
            import time
            time.sleep(pulse_s)
            GPIO.output(self.pin, GPIO.LOW if val==GPIO.HIGH else GPIO.HIGH)
        except Exception as e:
            print('GPIO error:', e)
    def cleanup(self):
        if ON_PI:
            GPIO.cleanup()