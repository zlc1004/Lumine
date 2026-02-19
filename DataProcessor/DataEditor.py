import sys
import os
import json
import pygame
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


class DataEditor:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata.jsonl"
        self.frames_dir = self.dataset_path / "frames"

        if not self.metadata_path.exists():
            messagebox.showerror(
                "Error", f"metadata.jsonl not found in {self.dataset_path}"
            )
            sys.exit(1)

        self.samples = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        if not self.samples:
            messagebox.showerror("Error", "No samples found in metadata")
            sys.exit(1)

        pygame.init()
        self.screen_width = 1280
        self.screen_height = 850
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Lumine Data Editor - {self.dataset_path.name}")

        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)

        self.current_idx = 0
        self.start_marker = 0
        self.end_marker = len(self.samples) - 1

        self.running = True
        self.clock = pygame.time.Clock()
        self.image_cache = {}

        # Hide the root tkinter window
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()

    def load_image(self, filename):
        if filename in self.image_cache:
            return self.image_cache[filename]

        path = self.frames_dir / filename
        if not path.exists():
            return None

        try:
            img = pygame.image.load(str(path))
            img = pygame.transform.scale(img, (1280, 720))
            self.image_cache[filename] = img

            # Simple cache management
            if len(self.image_cache) > 100:
                keys = list(self.image_cache.keys())
                del self.image_cache[keys[0]]
            return img
        except:
            return None

    def edit_text(self, field):
        sample = self.samples[self.current_idx]
        current_val = sample.get(field, "")

        # Use Tkinter dialog for text entry instead of console
        new_val = simpledialog.askstring(
            f"Edit {field.capitalize()}",
            f"Enter {field} for frame {self.current_idx}:",
            initialvalue=current_val,
        )

        if new_val is not None:
            if new_val.strip() == "" or new_val.lower() == "clear":
                sample.pop(field, None)
            else:
                sample[field] = new_val

    def save_dataset(self):
        new_path = self.dataset_path.parent / (self.dataset_path.name + "_edited")

        if messagebox.askyesno(
            "Save",
            f"Save edited dataset to {new_path.name}?\nRange: {self.start_marker} to {self.end_marker}",
        ):
            new_path.mkdir(parents=True, exist_ok=True)
            new_frames_dir = new_path / "frames"
            new_frames_dir.mkdir(parents=True, exist_ok=True)

            new_samples = self.samples[self.start_marker : self.end_marker + 1]

            # Copy and update
            with open(new_path / "metadata.jsonl", "w", encoding="utf-8") as f:
                for i, sample in enumerate(new_samples):
                    sample_copy = sample.copy()
                    sample_copy["frame_idx"] = i
                    f.write(json.dumps(sample_copy, ensure_ascii=False) + "\n")

                    src = self.frames_dir / sample["image"]
                    dst = new_frames_dir / sample["image"]
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)

            # Update stage files
            with open(new_path / "stage1_pretrain.jsonl", "w", encoding="utf-8") as f:
                for sample in new_samples:
                    out = {"image": sample["image"], "text": sample["action"]}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

            messagebox.showinfo(
                "Success", f"Dataset saved successfully to {new_path.name}"
            )

    def run(self):
        while self.running:
            sample = self.samples[self.current_idx]
            img = self.load_image(sample["image"])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.current_idx = min(
                            self.current_idx + 1, len(self.samples) - 1
                        )
                    elif event.key == pygame.K_LEFT:
                        self.current_idx = max(self.current_idx - 1, 0)
                    elif event.key == pygame.K_PAGEDOWN:
                        self.current_idx = min(
                            self.current_idx + 50, len(self.samples) - 1
                        )
                    elif event.key == pygame.K_PAGEUP:
                        self.current_idx = max(self.current_idx - 50, 0)
                    elif event.key == pygame.K_s:
                        self.start_marker = self.current_idx
                    elif event.key == pygame.K_e:
                        self.end_marker = self.current_idx
                    elif event.key == pygame.K_i:
                        self.edit_text("instruction")
                    elif event.key == pygame.K_t:
                        self.edit_text("thought")
                    elif event.key == pygame.K_RETURN:
                        if pygame.key.get_mods() & pygame.KMOD_CTRL:
                            self.save_dataset()
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if (
                        event.pos[1] > 720 and event.pos[1] < 735
                    ):  # Click on progress bar
                        self.current_idx = int(
                            (event.pos[0] / self.screen_width) * (len(self.samples) - 1)
                        )

            self.screen.fill((20, 20, 20))

            if img:
                self.screen.blit(img, (0, 0))

            # UI Panels
            pygame.draw.rect(self.screen, (40, 40, 40), (0, 720, 1280, 130))

            # Progress Bar
            pygame.draw.rect(self.screen, (60, 60, 60), (0, 720, 1280, 15))
            # Current pos
            px = int((self.current_idx / (len(self.samples) - 1)) * 1280)
            pygame.draw.rect(self.screen, (200, 0, 0), (px - 2, 720, 4, 15))

            # Selection Range
            sx = int((self.start_marker / (len(self.samples) - 1)) * 1280)
            ex = int((self.end_marker / (len(self.samples) - 1)) * 1280)
            pygame.draw.rect(self.screen, (0, 150, 0, 100), (sx, 720, ex - sx, 15))
            pygame.draw.rect(self.screen, (0, 255, 0), (sx, 720, 2, 15))
            pygame.draw.rect(self.screen, (0, 255, 0), (ex, 720, 2, 15))

            # Info text
            info_y = 745
            self.screen.blit(
                self.font.render(
                    f"Frame: {self.current_idx}/{len(self.samples) - 1}",
                    True,
                    (255, 255, 255),
                ),
                (10, info_y),
            )
            self.screen.blit(
                self.font.render(f"File: {sample['image']}", True, (200, 200, 200)),
                (10, info_y + 20),
            )

            action_text = sample["action"]
            if len(action_text) > 80:
                action_text = action_text[:77] + "..."
            self.screen.blit(
                self.font.render(f"Action: {action_text}", True, (0, 255, 100)),
                (10, info_y + 40),
            )

            # Inst/Thought display
            inst = sample.get("instruction", "[None]")
            thought = sample.get("thought", "[None]")
            self.screen.blit(
                self.font.render(f"Instruction (I): {inst}", True, (255, 255, 0)),
                (10, info_y + 60),
            )
            self.screen.blit(
                self.font.render(f"Thought (T): {thought}", True, (0, 200, 255)),
                (10, info_y + 80),
            )

            # Controls help
            help_x = 900
            controls = [
                "Left/Right: Nav",
                "PgUp/Dn: Fast Nav",
                "S: Set Start / E: Set End",
                "I: Edit Instruction",
                "T: Edit Thought",
                "Ctrl+Enter: SAVE EXPORT",
            ]
            for i, line in enumerate(controls):
                self.screen.blit(
                    self.font.render(line, True, (180, 180, 180)),
                    (help_x, info_y + i * 18),
                )

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        self.tk_root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askdirectory(
        title="Select Dataset Folder (containing metadata.jsonl)"
    )

    if path:
        DataEditor(path).run()
    else:
        print("No folder selected.")
