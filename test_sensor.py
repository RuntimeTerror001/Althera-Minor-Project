from dashboard_bridge import SensorReader
import time

print("Available ports:", SensorReader.list_all_ports())
port = SensorReader.find_serial_port()
print("Detected port:", port)

reader = SensorReader()
print(f"Connecting to {port}...")
ok, err = reader.connect_blocking(port=port, timeout=10)

print("Connect ok?", ok)
if err:
    print("Error:", err)

if ok:
    for i in range(10):
        print("Healthy?", reader.is_healthy(), "Latest:", reader.get_latest())
        time.sleep(1)

reader.stop()
