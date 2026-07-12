from __future__ import annotations

import pyxel

from src.bulletml_runtime import BulletState

PINK_BULLET_COLOR = 14
BLUE_BULLET_COLOR = 12
LIGHT_BLUE_BULLET_COLOR = 6
BLUE_SPEED_THRESHOLD = 2.0
BULLET_RADIUS = 1
VIEWPORT_MARGIN = 24.0


def draw_bullet(bullet: BulletState) -> None:
    if bullet.speed < BLUE_SPEED_THRESHOLD:
        color = PINK_BULLET_COLOR
    else:
        spoke_index = round(bullet.direction / 24.0)
        color = (
            BLUE_BULLET_COLOR
            if spoke_index % 2 == 0
            else LIGHT_BLUE_BULLET_COLOR
        )
    pyxel.circ(
        int(bullet.x),
        int(bullet.y),
        BULLET_RADIUS,
        color,
    )


def is_outside_viewport(
    bullet: BulletState,
    *,
    width: int,
    height: int,
    margin: float = VIEWPORT_MARGIN,
) -> bool:
    return (
        bullet.x < -margin
        or bullet.x > width + margin
        or bullet.y < -margin
        or bullet.y > height + margin
    )
