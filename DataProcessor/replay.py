import sys
import time
import ctypes
from ctypes import wintypes

# WinAPI constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000

KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002

user32 = ctypes.windll.user32


def read_log_file(filepath):
    events = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    timestamp = int(parts[0])
                    event_type = parts[1]
                    data = parts[2:] if len(parts) > 2 else []
                    events.append((timestamp, event_type, data))
                except ValueError:
                    continue
    return events


VK_MAP = {
    "Esc": 0x1B,
    "Tab": 0x09,
    "Caps": 0x14,
    "Shift": 0x10,
    "Ctrl": 0x11,
    "Alt": 0x12,
    "Space": 0x20,
    "Back": 0x08,
    "Enter": 0x0D,
    "Left": 0x25,
    "Up": 0x26,
    "Right": 0x27,
    "Down": 0x28,
    "Home": 0x24,
    "End": 0x23,
    "PgUp": 0x21,
    "PgDn": 0x22,
    "Ins": 0x2D,
    "Del": 0x2E,
    "One": 0x70,
    "Two": 0x71,
    "Three": 0x72,
    "Four": 0x73,
    "Five": 0x74,
    "Six": 0x75,
    "Seven": 0x76,
    "Eight": 0x77,
    "Nine": 0x78,
    "Ten": 0x79,
    "Eleven": 0x7A,
    "Twelve": 0x7B,
}

for i, name in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
):
    VK_MAP[name] = ord(str(i))

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    VK_MAP[letter] = ord(letter)

for i in range(10):
    VK_MAP[f"Num{i}"] = 0x60 + i
VK_MAP["Num*"] = 0x6A
VK_MAP["Num+"] = 0x6B
VK_MAP["Num-"] = 0x6D
VK_MAP["Num."] = 0x6E
VK_MAP["Num/"] = 0x6F

MOUSE_BTN_MAP = {
    "LB": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "RB": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "MB": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


def send_key(token, is_down):
    vk = VK_MAP.get(token)
    if vk is not None:
        flags = KEYEVENTF_KEYDOWN if is_down else KEYEVENTF_KEYUP
        user32.keybd_event(vk, 0, flags, 0)


def send_mouse_button(token, is_down):
    flags = MOUSE_BTN_MAP.get(token)
    if flags:
        user32.mouse_event(flags[0] if is_down else flags[1], 0, 0, 0, 0)


def send_mouse_wheel(delta):
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(delta), 0)


def send_mouse_abs(x, y):
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    fx = int((x * 65535) / (screen_width - 1))
    fy = int((y * 65535) / (screen_height - 1))
    user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, fx, fy, 0, 0)


def send_mouse_rel(dx, dy):
    user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)


class Replay:
    def __init__(self, filepath, speed=1.0, loop=True):
        self.filepath = filepath
        self.events = read_log_file(filepath)
        if not self.events:
            print("No events found in log file")
            sys.exit(1)

        self.speed = speed
        self.loop = loop
        self.first_timestamp = self.events[0][0]
        self.running = True
        self.held_keys = set()

    def release_all(self):
        for token in list(self.held_keys):
            if token in MOUSE_BTN_MAP:
                send_mouse_button(token, False)
            else:
                send_key(token, False)
        self.held_keys.clear()

    def run(self):
        print(f"Replaying: {self.filepath}")
        print(f"Speed: {self.speed}x, Loop: {self.loop}")
        print("Press Ctrl+C to stop...")

        try:
            while self.running:
                start_time = time.perf_counter()
                self.release_all()

                for timestamp, event_type, data in self.events:
                    if not self.running:
                        break

                    rel_time_filetime = timestamp - self.first_timestamp
                    rel_time_sec = rel_time_filetime / 10_000_000
                    wait_time = rel_time_sec / self.speed

                    while True:
                        current_time = time.perf_counter()
                        elapsed = current_time - start_time

                        if elapsed >= wait_time:
                            break

                        time.sleep(0.001)

                    if event_type == "KEY_CHUNK":
                        new_held = set(data[0].split()) if data else set()

                        # Keys to press
                        for token in new_held - self.held_keys:
                            if token in MOUSE_BTN_MAP:
                                send_mouse_button(token, True)
                            else:
                                send_key(token, True)

                        # Keys to release
                        for token in self.held_keys - new_held:
                            if token in MOUSE_BTN_MAP:
                                send_mouse_button(token, False)
                            else:
                                send_key(token, False)

                        self.held_keys = new_held

                    elif event_type == "MOUSE" and len(data) >= 2:
                        if data[0] == "WHEEL":
                            try:
                                send_mouse_wheel(int(data[1]))
                            except ValueError:
                                pass

                    elif event_type == "MOUSE_ABS" and len(data) >= 2:
                        try:
                            send_mouse_abs(int(data[0]), int(data[1]))
                        except ValueError:
                            pass

                    elif event_type == "MOUSE_REL" and len(data) >= 2:
                        try:
                            send_mouse_rel(int(data[0]), int(data[1]))
                        except ValueError:
                            pass

                if self.loop and self.running:
                    print("Looping...")
                else:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.release_all()

        print("Replay finished.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <log_file> [speed] [no-loop]")
        sys.exit(1)

    filepath = sys.argv[1]
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    loop = "no-loop" not in sys.argv

    Replay(filepath, speed, loop).run()
