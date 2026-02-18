import sys
import time
import pygame


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


class Overlay:
    def __init__(self, filepath):
        self.filepath = filepath
        self.events = read_log_file(filepath)
        if not self.events:
            print("No events found in log file")
            sys.exit(1)

        self.first_timestamp = self.events[0][0]
        self.program_start = time.time()
        self.offset = self.first_timestamp
        self.rel_x, self.rel_y = 0, 0
        pygame.init()
        pygame.display.set_caption("Mouse Recorder Overlay")

        info = pygame.display.Info()
        self.screen = pygame.display.set_mode(
            (info.current_w, info.current_h),
            pygame.NOFRAME | pygame.SCALED,
        )

        self.clock = pygame.time.Clock()
        self.running = True

        self.center = (info.current_w // 2, info.current_h // 2)
        self.grid_pos = list(self.center)
        self.cursor_locked = False

        self.font = pygame.font.Font(None, 36)
        self.grid_spacing = 50

        self.set_layered_window()
        self.always_on_top()

    def always_on_top(self):
        import ctypes

        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE
        )

    def set_layered_window(self):
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        LWA_ALPHA = 0x2
        hwnd = pygame.display.get_wm_info()["window"]
        ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)

    def get_current_state(self):
        elapsed_sec = time.time() - self.program_start
        elapsed_filetime = int(elapsed_sec * 10_000_000)
        current_timestamp = self.offset + elapsed_filetime

        x, y = None, None

        for timestamp, event_type, data in self.events:
            if timestamp > current_timestamp:
                break
            if event_type == "MOUSE":
                if len(data) > 0:
                    if data[0] == "LOCK":
                        self.cursor_locked = True
                        self.rel_x, self.rel_y = 0, 0
                    elif data[0] == "UNLOCK":
                        self.cursor_locked = False
            elif event_type == "MOUSE_ABS" and len(data) >= 2:
                try:
                    x = int(data[0])
                    y = int(data[1])
                except ValueError:
                    pass
            elif event_type == "MOUSE_RAW_REL" and len(data) >= 2:
                try:
                    self.rel_x += int(data[0])
                    self.rel_y += int(data[1])
                except ValueError:
                    pass
            elif event_type == "MOUSE_REL" and len(data) >= 2:
                try:
                    self.rel_x += int(data[0])
                    self.rel_y += int(data[1])
                except ValueError:
                    pass

        is_aim_mode = self.cursor_locked

        if is_aim_mode:
            if self.rel_x != 0 or self.rel_y != 0:
                self.grid_pos = [self.rel_x, self.rel_y]
            return self.grid_pos, is_aim_mode
        else:
            if x is not None and y is not None:
                return (x, y), is_aim_mode
            return self.center, is_aim_mode

    def get_display_time(self):
        elapsed = time.time() - self.program_start
        secs = int(elapsed)
        mins = secs // 60
        secs = secs % 60
        ms = int((elapsed % 1) * 1000)
        return f"{mins:02d}:{secs:02d}.{ms:03d}"

    def draw_grid(self, pos):
        w, h = self.screen.get_size()
        spacing = self.grid_spacing

        offset_x = pos[0] % spacing
        offset_y = pos[1] % spacing

        for x in range(offset_x, w, spacing):
            pygame.draw.line(self.screen, (80, 80, 80), (x, 0), (x, h), 1)
        for y in range(offset_y, h, spacing):
            pygame.draw.line(self.screen, (80, 80, 80), (0, y), (w, y), 1)

        cross_x = pos[0] - (pos[0] % spacing)
        cross_y = pos[1] - (pos[1] % spacing)
        pygame.draw.line(
            self.screen,
            (0, 255, 0),
            (cross_x - 15, cross_y),
            (cross_x + 15, cross_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            (0, 255, 0),
            (cross_x, cross_y - 15),
            (cross_x, cross_y + 15),
            2,
        )

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            pos, is_aim_mode = self.get_current_state()

            self.screen.fill((0, 0, 0, 0))

            if is_aim_mode:
                self.draw_grid(pos)
                mode_text = "AIM (REL)"
            else:
                pygame.draw.circle(self.screen, (255, 0, 0), pos, 20, 3)
                mode_text = "MENU (ABS)"

            time_text = self.get_display_time()
            text_surface = self.font.render(
                f"{time_text} | {mode_text}", True, (255, 255, 255)
            )
            text_rect = text_surface.get_rect(topleft=(10, 10))

            bg_rect = text_rect.inflate(10, 5)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            pygame.draw.rect(self.screen, (255, 0, 0), bg_rect, 1)
            self.screen.blit(text_surface, text_rect)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <log_file>")
        sys.exit(1)

    Overlay(sys.argv[1]).run()
