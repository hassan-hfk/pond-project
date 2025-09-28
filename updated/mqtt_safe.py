# mqtt_safe.py
import paho.mqtt.client as mqtt
import threading
import time
import json

class MQTTSafeClient:
    def __init__(self, broker="localhost", port=1883, client_id="child_monitor"):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client(client_id)
        self.connected = False

        # Bind callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        # Start async connection
        self.client.connect_async(self.broker, self.port, 60)
        self.client.loop_start()  # runs network loop in background

        # Lock for thread-safe publishing
        self.lock = threading.Lock()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to broker {self.broker}:{self.port}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("[MQTT] Disconnected")

    def publish(self, topic, payload):
        """Thread-safe publish, never blocks main loop."""
        try:
            with self.lock:
                self.client.publish(topic, json.dumps(payload), qos=0)
        except Exception as e:
            print(f"[MQTT] Publish failed: {e}")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
