import serial
import time

# Configure the serial port
# Replace 'COMx' with your actual serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
# Set the baudrate to match your device's configuration
ser = serial.Serial(
    port='COMx',  # Change to your serial port
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1 # Timeout for read operations
)

print(f"Opening serial port {ser.name}...")

try:
    # Open the serial port
    if not ser.isOpen():
        ser.open()

    print("Serial port opened successfully.")

    # Data to send
    data_to_send = "Hello, UART!"
    print(f"Sending: '{data_to_send}'")

    # Encode the string to bytes before sending
    ser.write(data_to_send.encode('utf-8'))

    # Optional: Read response if expected
    # time.sleep(0.1) # Give some time for the device to respond
    # if ser.in_waiting > 0:
    #     received_data = ser.readline().decode('utf-8').strip()
    #     print(f"Received: '{received_data}'")

except serial.SerialException as e:
    print(f"Error opening or communicating with serial port: {e}")

finally:
    # Close the serial port
    if ser.isOpen():
        ser.close()
        print("Serial port closed.")