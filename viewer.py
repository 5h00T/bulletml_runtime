from __future__ import annotations

from pathlib import Path

import pyxel

from src.bulletml_parser import BulletMLParseError, load_bulletml
from src.bulletml_runtime import BulletMLRuntime, BulletMLRuntimeError
from pyxel_bullet_view import draw_bullet, is_outside_viewport

ROOT = Path(__file__).resolve().parent
SIMPLE_BARRAGE_PATH = ROOT / "simple_barrage.xml"
WINDOW_WIDTH = 320
WINDOW_HEIGHT = 240
HEADER_HEIGHT = 36
EMITTER_WIDTH = 6
EMITTER_HEIGHT = 4
PLAYER_SIZE = 5
PLAYER_SPEED = 2.0
PLAYER_SLOW_SPEED = 1.0
DIAGONAL_FACTOR = 2**-0.5


def runtime_positions(
    document_type: str,
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    playfield_center_y = (HEADER_HEIGHT + height) / 2.0
    if document_type == "horizontal":
        return (float(width - 12), playfield_center_y), (12.0, playfield_center_y)
    return (
        (width / 2.0, float(HEADER_HEIGHT + 28)),
        (width / 2.0, float(height - 12)),
    )


def move_player_position(
    position: tuple[float, float],
    direction_x: int,
    direction_y: int,
    speed: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    movement_speed = speed
    if direction_x != 0 and direction_y != 0:
        movement_speed *= DIAGONAL_FACTOR

    half_size = PLAYER_SIZE / 2.0
    x = position[0] + direction_x * movement_speed
    y = position[1] + direction_y * movement_speed
    return (
        min(max(x, half_size), width - half_size),
        min(max(y, HEADER_HEIGHT + half_size), height - half_size),
    )


def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return f"{text[: max_length - 3]}..."


class SimpleBarrageApp:
    def __init__(self) -> None:
        pyxel.init(WINDOW_WIDTH, WINDOW_HEIGHT, title="BulletML Simple Barrage")
        self.error_message: str | None = None
        self.runtime: BulletMLRuntime | None = None
        self.emitter_position = (pyxel.width / 2.0, float(HEADER_HEIGHT + 12))
        self.player_position = (pyxel.width / 2.0, float(pyxel.height - 12))

        self._load_runtime()
        pyxel.run(self.update, self.draw)

    def _load_runtime(self) -> None:
        try:
            document = load_bulletml(SIMPLE_BARRAGE_PATH)
            self.emitter_position, self.player_position = runtime_positions(
                document.type,
                pyxel.width,
                pyxel.height,
            )
            self.runtime = BulletMLRuntime(
                document,
                origin_x=self.emitter_position[0],
                origin_y=self.emitter_position[1],
                target_position=lambda: self.player_position,
            )
            self.error_message = None
        except (BulletMLParseError, BulletMLRuntimeError, OSError) as exc:
            self.runtime = None
            self.error_message = str(exc)

    def update(self) -> None:
        if pyxel.btnp(pyxel.KEY_R):
            self._load_runtime()
        movement_speed = (
            PLAYER_SLOW_SPEED if pyxel.btn(pyxel.KEY_SHIFT) else PLAYER_SPEED
        )
        direction_x = int(pyxel.btn(pyxel.KEY_D)) - int(pyxel.btn(pyxel.KEY_A))
        direction_y = int(pyxel.btn(pyxel.KEY_S)) - int(pyxel.btn(pyxel.KEY_W))
        self.player_position = move_player_position(
            self.player_position,
            direction_x,
            direction_y,
            movement_speed,
            pyxel.width,
            pyxel.height,
        )
        if self.runtime is not None:
            self.runtime.step()
            self.runtime.remove_bullets_if(
                lambda bullet: is_outside_viewport(
                    bullet,
                    width=pyxel.width,
                    height=pyxel.height,
                )
            )

    def draw(self) -> None:
        pyxel.cls(1)
        pyxel.rect(0, 0, pyxel.width, HEADER_HEIGHT, 0)
        pyxel.rect(0, HEADER_HEIGHT, pyxel.width, pyxel.height - HEADER_HEIGHT, 1)
        pyxel.rect(
            int(self.emitter_position[0]) - EMITTER_WIDTH // 2,
            int(self.emitter_position[1]) - EMITTER_HEIGHT // 2,
            EMITTER_WIDTH,
            EMITTER_HEIGHT,
            11,
        )
        pyxel.rect(
            int(self.player_position[0]) - PLAYER_SIZE // 2,
            int(self.player_position[1]) - PLAYER_SIZE // 2,
            PLAYER_SIZE,
            PLAYER_SIZE,
            8,
        )
        pyxel.pset(int(self.player_position[0]), int(self.player_position[1]), 7)

        pyxel.text(4, 4, f"xml: {SIMPLE_BARRAGE_PATH.name}", 7)

        bullets = () if self.runtime is None else self.runtime.bullets
        frame_count = 0 if self.runtime is None else self.runtime.frame_count
        bullet_count = len(bullets)
        pyxel.text(4, 12, f"frame: {frame_count}", 6)
        pyxel.text(4, 20, f"bullets: {bullet_count}", 10)
        pyxel.text(4, 28, "WASD: move  SHIFT: slow  R: reload", 12)

        if self.runtime is not None:
            for bullet in bullets:
                draw_bullet(bullet)
            return

        pyxel.text(4, 48, "runtime load failed", 8)
        if self.error_message is not None:
            pyxel.text(4, 56, truncate_text(self.error_message, 31), 7)


def main() -> None:
    SimpleBarrageApp()


if __name__ == "__main__":
    main()
