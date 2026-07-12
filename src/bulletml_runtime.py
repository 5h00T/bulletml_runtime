from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from math import atan2, cos, degrees, radians, sin
from typing import Final

from .bulletml_expression import BulletMLExpressionError, evaluate_expression
from .bulletml_parser import (
    Accel,
    Action,
    ActionRef,
    Bullet,
    BulletMLDocument,
    BulletRef,
    ChangeDirection,
    ChangeSpeed,
    Direction,
    Fire,
    FireRef,
    Horizontal,
    Repeat,
    Speed,
    Vanish,
    Vertical,
    Wait,
    action_index,
    bullet_index,
    fire_index,
)

# top action が明示されないサンプルでは、BulletML でよく使われる top を起点にする。
_DEFAULT_ROOT_LABEL: Final = "top"


class BulletMLRuntimeError(RuntimeError):
    # パース済み BulletML を実行できない場合に使うランタイム例外。
    pass


@dataclass(slots=True)
class ActionFrame:
    # action の実行位置、位置引数、repeat の残り回数、wait の状態を保持する。
    action: Action
    parameters: tuple[float, ...] = ()
    repeat_remaining: int = 1
    program_counter: int = 0
    wait_frames: int = 0


def _new_action_frames() -> list[ActionFrame]:
    # dataclass の mutable default を避けるため、空のフレームスタックを都度作る。
    return []


@dataclass(slots=True)
class ActionThread:
    # action スタック、sequence 用の直前値、継続中の移動変化を保持する。
    frames: list[ActionFrame] = field(default_factory=_new_action_frames)
    last_fire_direction: float = 0.0
    last_fire_speed: float = 1.0
    speed_change_frames: int = 0
    speed_change_delta: float = 0.0
    speed_change_target: float | None = None
    direction_change_frames: int = 0
    direction_change_delta: float = 0.0
    direction_change_target: float | None = None
    direction_change_aim_offset: float | None = None
    accel_frames: int = 0
    horizontal_accel_delta: float = 0.0
    horizontal_accel_target: float | None = None
    vertical_accel_delta: float = 0.0
    vertical_accel_target: float | None = None


def _new_action_threads() -> list[ActionThread]:
    # RunnerState ごとに独立した action スレッドリストを作る。
    return []


@dataclass(slots=True)
class RunnerState:
    # 発射元と弾に共通する位置、移動方向、速度、生存状態を表す。
    x: float
    y: float
    direction: float = 0.0
    speed: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    alive: bool = True
    threads: list[ActionThread] = field(default_factory=_new_action_threads)


@dataclass(slots=True)
class BulletState(RunnerState):
    # 発射元と区別して、外部へ弾の実行状態だけを公開する。
    pass


class BulletMLRuntime:
    def __init__(
        self,
        document: BulletMLDocument,
        *,
        origin_x: float,
        origin_y: float,
        target_position: Callable[[], tuple[float, float]],
        root_action_label: str = _DEFAULT_ROOT_LABEL,
        random_func: Callable[[], float] = random.random,
        rank: float = 0.5,
    ) -> None:
        self.document = document
        self.frame_count = 0
        self._bullets: list[BulletState] = []
        self._random_func = random_func
        self.rank = float(rank)
        self.target_position = target_position

        self._actions = action_index(document)
        self._bullet_definitions = bullet_index(document)
        self._fire_definitions = fire_index(document)
        self._validate_supported_document(document)

        # 参考デモと同様に、既定の top では top で始まる全 action を並行実行する。
        # 明示的なラベルが渡された場合は、その action だけを開始する。
        if root_action_label == _DEFAULT_ROOT_LABEL:
            root_actions = tuple(
                action
                for action in document.actions
                if action.label is not None
                and action.label.startswith(_DEFAULT_ROOT_LABEL)
            )
        else:
            root_action = self._actions.get(root_action_label)
            root_actions = () if root_action is None else (root_action,)

        if not root_actions:
            if len(document.actions) == 1:
                root_actions = (document.actions[0],)
            else:
                raise BulletMLRuntimeError(
                    f"top-level action '{root_action_label}' is required"
                )

        self._emitter = RunnerState(
            x=origin_x,
            y=origin_y,
            threads=[
                ActionThread(
                    frames=[
                        ActionFrame(action=root_action),
                    ]
                )
                for root_action in root_actions
            ],
        )

    @property
    def bullets(self) -> tuple[BulletState, ...]:
        # 外部から内部リストを直接変更されないよう、読み取り用の tuple を返す。
        return tuple(self._bullets)

    def remove_bullets_if(
        self,
        predicate: Callable[[BulletState], bool],
    ) -> None:
        # 画面外判定や当たり判定など、ホスト側の規則で指定された弾を除去する。
        for bullet in self._bullets:
            if predicate(bullet):
                bullet.alive = False
        self._remove_dead_bullets()

    def step(self) -> None:
        # 1 フレーム進める。新しく生まれた弾は次フレームから移動対象にする。
        self.frame_count += 1
        active_bullets = list(self._bullets)

        # 発射元の action を先に進めて、このフレームで必要な弾を生成する。
        if self._emitter.alive:
            self._step_runner(self._emitter)

        # 既存の弾は action 実行と移動だけを行い、画面固有の寿命管理はしない。
        for bullet in active_bullets:
            if not bullet.alive:
                continue
            self._step_runner(bullet)
            if bullet.alive:
                self._move_bullet(bullet)

        # BulletML の vanish で死亡した弾をリストから取り除く。
        self._remove_dead_bullets()

    def _remove_dead_bullets(self) -> None:
        self._bullets = [bullet for bullet in self._bullets if bullet.alive]

    def _step_runner(self, runner: RunnerState) -> None:
        # Runner が持つ各 action スレッドを進め、完了したスレッドを掃除する。
        for thread in list(runner.threads):
            self._step_thread(runner, thread)
        runner.threads = [
            thread
            for thread in runner.threads
            if (
                thread.frames
                or thread.speed_change_frames > 0
                or thread.direction_change_frames > 0
                or thread.accel_frames > 0
            )
        ]

    def _step_thread(self, runner: RunnerState, thread: ActionThread) -> None:
        # スタックトップの ActionFrame を、wait や repeat を考慮しながら実行する。
        if not runner.alive:
            return

        self._update_speed_change(runner, thread)
        self._update_direction_change(runner, thread)
        self._update_accel(runner, thread)

        if not thread.frames:
            return

        frame = thread.frames[-1]
        if frame.wait_frames > 0:
            # wait 中はカウンタを 1 つ減らして、このフレームの命令実行を止める。
            frame.wait_frames -= 1
            return

        while thread.frames and runner.alive:
            frame = thread.frames[-1]
            if frame.program_counter >= len(frame.action.children):
                # action 末尾では repeat を巻き戻し、残りがなければスタックを抜く。
                if frame.repeat_remaining > 1:
                    frame.repeat_remaining -= 1
                    frame.program_counter = 0
                    continue
                thread.frames.pop()
                continue

            command = frame.action.children[frame.program_counter]
            frame.program_counter += 1

            if isinstance(command, Fire):
                # fire は現在の Runner の位置から新しい弾を生成する。
                self._spawn_fire(command, runner, thread, frame.parameters)
                continue

            if isinstance(command, FireRef):
                # 参照先 fire を解決し、評価済み param を発射スコープにする。
                referenced_fire, fire_parameters = self._resolve_fire(
                    command,
                    frame.parameters,
                )
                self._spawn_fire(
                    referenced_fire,
                    runner,
                    thread,
                    fire_parameters,
                )
                continue

            if isinstance(command, ChangeSpeed):
                self._start_speed_change(
                    command,
                    runner,
                    thread,
                    frame.parameters,
                )
                continue

            if isinstance(command, ChangeDirection):
                self._start_direction_change(
                    command,
                    runner,
                    thread,
                    frame.parameters,
                )
                continue

            if isinstance(command, Accel):
                self._start_accel(
                    command,
                    runner,
                    thread,
                    frame.parameters,
                )
                continue

            if isinstance(command, Wait):
                # wait は整数フレームに丸め、0 以下ならそのまま次の命令へ進む。
                wait_frames = max(
                    0,
                    self._evaluate_int(command.expression, frame.parameters),
                )
                if wait_frames > 0:
                    frame.wait_frames = wait_frames
                    return
                continue

            if isinstance(command, Repeat):
                # repeat は子 action を新しいフレームとして積み、指定回数だけ実行する。
                repeat_times = max(
                    0,
                    self._evaluate_int(
                        command.times.expression,
                        frame.parameters,
                    ),
                )
                if repeat_times == 0:
                    continue
                repeat_action, repeat_parameters = self._resolve_action(
                    command.action,
                    frame.parameters,
                )
                thread.frames.append(
                    ActionFrame(
                        action=repeat_action,
                        parameters=repeat_parameters,
                        repeat_remaining=repeat_times,
                    )
                )
                continue

            if isinstance(command, Action):
                # inline action は呼び出しのようにスタックへ積んで実行する。
                thread.frames.append(
                    ActionFrame(action=command, parameters=frame.parameters)
                )
                continue

            if isinstance(command, ActionRef):
                # 参照先 action を解決し、評価済み param を新しいスコープにする。
                referenced_action, parameters = self._resolve_action(
                    command,
                    frame.parameters,
                )
                thread.frames.append(
                    ActionFrame(action=referenced_action, parameters=parameters)
                )
                continue

            # ここまでの型分岐に該当しない残りの命令は Vanish。
            runner.alive = False
            thread.frames.clear()
            return

    def _update_speed_change(
        self,
        runner: RunnerState,
        thread: ActionThread,
    ) -> None:
        if thread.speed_change_frames <= 0:
            return

        runner.speed += thread.speed_change_delta
        thread.speed_change_frames -= 1
        if (
            thread.speed_change_frames == 0
            and thread.speed_change_target is not None
        ):
            runner.speed = thread.speed_change_target
            thread.speed_change_delta = 0.0
            thread.speed_change_target = None

    def _start_speed_change(
        self,
        command: ChangeSpeed,
        runner: RunnerState,
        thread: ActionThread,
        parameters: tuple[float, ...],
    ) -> None:
        term = max(0, self._evaluate_int(command.term.expression, parameters))
        value = self._evaluate_float(command.speed.expression, parameters)

        if command.speed.type == "sequence":
            thread.speed_change_frames = term
            thread.speed_change_delta = value
            thread.speed_change_target = None
            return

        if command.speed.type == "absolute":
            target = value
        elif command.speed.type == "relative":
            target = runner.speed + value
        else:
            raise BulletMLRuntimeError(
                f"unsupported changeSpeed type: {command.speed.type}"
            )

        if term == 0:
            runner.speed = target
            thread.speed_change_frames = 0
            thread.speed_change_delta = 0.0
            thread.speed_change_target = None
            return

        thread.speed_change_frames = term
        thread.speed_change_delta = (target - runner.speed) / term
        thread.speed_change_target = target

    def _update_direction_change(
        self,
        runner: RunnerState,
        thread: ActionThread,
    ) -> None:
        if thread.direction_change_frames <= 0:
            return

        if thread.direction_change_aim_offset is not None:
            target = (
                self._aim_direction(runner) + thread.direction_change_aim_offset
            )
            difference = self._shortest_direction_delta(runner.direction, target)
            runner.direction += difference / thread.direction_change_frames
            thread.direction_change_frames -= 1
            if thread.direction_change_frames == 0:
                thread.direction_change_delta = 0.0
                thread.direction_change_target = None
                thread.direction_change_aim_offset = None
            return

        runner.direction += thread.direction_change_delta
        thread.direction_change_frames -= 1
        if (
            thread.direction_change_frames == 0
            and thread.direction_change_target is not None
        ):
            runner.direction = thread.direction_change_target
            thread.direction_change_delta = 0.0
            thread.direction_change_target = None
            thread.direction_change_aim_offset = None

    def _start_direction_change(
        self,
        command: ChangeDirection,
        runner: RunnerState,
        thread: ActionThread,
        parameters: tuple[float, ...],
    ) -> None:
        term = max(0, self._evaluate_int(command.term.expression, parameters))
        value = self._evaluate_float(command.direction.expression, parameters)

        if command.direction.type == "sequence":
            thread.direction_change_frames = term
            thread.direction_change_delta = value
            thread.direction_change_target = None
            thread.direction_change_aim_offset = None
            return

        if command.direction.type == "absolute":
            target = value
        elif command.direction.type == "relative":
            target = runner.direction + value
        elif command.direction.type == "aim":
            if term > 0:
                thread.direction_change_frames = term
                thread.direction_change_delta = 0.0
                thread.direction_change_target = None
                thread.direction_change_aim_offset = value
                return
            target = self._aim_direction(runner) + value
        else:
            raise BulletMLRuntimeError(
                f"unsupported changeDirection type: {command.direction.type}"
            )

        difference = self._shortest_direction_delta(runner.direction, target)
        resolved_target = runner.direction + difference
        if term == 0:
            runner.direction = resolved_target
            thread.direction_change_frames = 0
            thread.direction_change_delta = 0.0
            thread.direction_change_target = None
            thread.direction_change_aim_offset = None
            return

        thread.direction_change_frames = term
        thread.direction_change_delta = difference / term
        thread.direction_change_target = resolved_target
        thread.direction_change_aim_offset = None

    @staticmethod
    def _shortest_direction_delta(current: float, target: float) -> float:
        difference = (target - current) % 360.0
        if difference > 180.0:
            difference -= 360.0
        return difference

    def _update_accel(
        self,
        runner: RunnerState,
        thread: ActionThread,
    ) -> None:
        if thread.accel_frames <= 0:
            return

        runner.velocity_x += thread.horizontal_accel_delta
        runner.velocity_y += thread.vertical_accel_delta
        thread.accel_frames -= 1
        if thread.accel_frames == 0:
            if thread.horizontal_accel_target is not None:
                runner.velocity_x = thread.horizontal_accel_target
            if thread.vertical_accel_target is not None:
                runner.velocity_y = thread.vertical_accel_target
            thread.horizontal_accel_delta = 0.0
            thread.horizontal_accel_target = None
            thread.vertical_accel_delta = 0.0
            thread.vertical_accel_target = None

    def _start_accel(
        self,
        command: Accel,
        runner: RunnerState,
        thread: ActionThread,
        parameters: tuple[float, ...],
    ) -> None:
        term = max(0, self._evaluate_int(command.term.expression, parameters))
        (
            runner.velocity_x,
            thread.horizontal_accel_delta,
            thread.horizontal_accel_target,
        ) = self._accel_axis_state(
            command.horizontal,
            runner.velocity_x,
            term,
            parameters,
        )
        (
            runner.velocity_y,
            thread.vertical_accel_delta,
            thread.vertical_accel_target,
        ) = self._accel_axis_state(
            command.vertical,
            runner.velocity_y,
            term,
            parameters,
        )
        thread.accel_frames = (
            term
            if command.horizontal is not None or command.vertical is not None
            else 0
        )

    def _accel_axis_state(
        self,
        axis: Horizontal | Vertical | None,
        current: float,
        term: int,
        parameters: tuple[float, ...],
    ) -> tuple[float, float, float | None]:
        if axis is None:
            return current, 0.0, None

        value = self._evaluate_float(axis.expression, parameters)
        if axis.type == "sequence":
            return current, value, None
        if axis.type == "absolute":
            target = value
        elif axis.type == "relative":
            target = current + value
        else:
            raise BulletMLRuntimeError(f"unsupported accel type: {axis.type}")

        if term == 0:
            return target, 0.0, None
        return current, (target - current) / term, target

    def _spawn_fire(
        self,
        fire: Fire,
        runner: RunnerState,
        thread: ActionThread,
        parameters: tuple[float, ...],
    ) -> None:
        bullet_definition, bullet_parameters = self._resolve_bullet(
            fire.bullet,
            parameters,
        )

        # fire 側の direction/speed があれば優先し、なければ bullet 側の値を使う。
        direction_source = fire.direction or bullet_definition.direction
        speed_source = fire.speed or bullet_definition.speed
        direction_parameters = (
            parameters if fire.direction is not None else bullet_parameters
        )
        speed_parameters = parameters if fire.speed is not None else bullet_parameters
        direction = self._resolve_direction(
            direction_source,
            runner,
            thread,
            direction_parameters,
        )
        speed = self._resolve_speed(
            speed_source,
            thread,
            speed_parameters,
        )

        # 生成した弾には bullet 内の action を独立したスレッドとして持たせる。
        bullet = BulletState(
            x=runner.x,
            y=runner.y,
            direction=direction,
            speed=speed,
            threads=self._build_threads(bullet_definition, bullet_parameters),
        )
        self._bullets.append(bullet)

        # sequence 型の次回発射が参照できるよう、直近の発射値を記録する。
        thread.last_fire_direction = direction
        thread.last_fire_speed = speed

    def _build_threads(
        self,
        bullet: Bullet,
        parameters: tuple[float, ...],
    ) -> list[ActionThread]:
        # bullet に定義された各 action から、弾専用の実行スレッドを作る。
        threads: list[ActionThread] = []
        for action in bullet.actions:
            resolved_action, resolved_parameters = self._resolve_action(
                action,
                parameters,
            )
            threads.append(
                ActionThread(
                    frames=[
                        ActionFrame(
                            action=resolved_action,
                            parameters=resolved_parameters,
                        )
                    ]
                )
            )
        return threads

    def _resolve_bullet(
        self,
        bullet: Bullet | BulletRef,
        caller_parameters: tuple[float, ...],
    ) -> tuple[Bullet, tuple[float, ...]]:
        if isinstance(bullet, Bullet):
            return bullet, caller_parameters

        referenced_bullet = self._get_referenced_bullet(bullet)
        parameters = tuple(
            self._evaluate_float(param.expression, caller_parameters)
            for param in bullet.params
        )
        return referenced_bullet, parameters

    def _resolve_fire(
        self,
        fire_ref: FireRef,
        caller_parameters: tuple[float, ...],
    ) -> tuple[Fire, tuple[float, ...]]:
        referenced_fire = self._get_referenced_fire(fire_ref)
        parameters = tuple(
            self._evaluate_float(param.expression, caller_parameters)
            for param in fire_ref.params
        )
        return referenced_fire, parameters

    def _get_referenced_fire(self, fire_ref: FireRef) -> Fire:
        fire = self._fire_definitions.get(fire_ref.label)
        if fire is None:
            raise BulletMLRuntimeError(
                f"fireRef label '{fire_ref.label}' was not found"
            )
        return fire

    def _get_referenced_bullet(self, bullet_ref: BulletRef) -> Bullet:
        bullet = self._bullet_definitions.get(bullet_ref.label)
        if bullet is None:
            raise BulletMLRuntimeError(
                f"bulletRef label '{bullet_ref.label}' was not found"
            )
        return bullet

    def _resolve_action(
        self,
        action: Action | ActionRef,
        caller_parameters: tuple[float, ...],
    ) -> tuple[Action, tuple[float, ...]]:
        if isinstance(action, Action):
            return action, caller_parameters

        referenced_action = self._get_referenced_action(action)
        parameters = tuple(
            self._evaluate_float(param.expression, caller_parameters)
            for param in action.params
        )
        return referenced_action, parameters

    def _get_referenced_action(self, action_ref: ActionRef) -> Action:
        action = self._actions.get(action_ref.label)
        if action is None:
            raise BulletMLRuntimeError(
                f"actionRef label '{action_ref.label}' was not found"
            )
        return action

    def _resolve_direction(
        self,
        direction: Direction | None,
        runner: RunnerState,
        thread: ActionThread,
        parameters: tuple[float, ...],
    ) -> float:
        # direction 省略時は BulletML の発射既定としてターゲットを狙う。
        if direction is None:
            return self._aim_direction(runner)

        value = self._evaluate_float(direction.expression, parameters)
        # absolute/relative/sequence の意味に従って角度を解決する。
        if direction.type == "absolute":
            return value
        if direction.type == "relative":
            return runner.direction + value
        if direction.type == "sequence":
            return thread.last_fire_direction + value
        if direction.type == "aim":
            return self._aim_direction(runner) + value
        raise BulletMLRuntimeError(
            f"direction type '{direction.type}' is not supported by the minimal runtime"
        )

    def _aim_direction(self, runner: RunnerState) -> float:
        target_x, target_y = self.target_position()
        delta_x = float(target_x) - runner.x
        delta_y = float(target_y) - runner.y
        # BulletML 参考実装の角度系は 0 度が上、90 度が右。
        return degrees(atan2(delta_x, -delta_y)) % 360.0

    def _resolve_speed(
        self,
        speed: Speed | None,
        thread: ActionThread,
        parameters: tuple[float, ...],
    ) -> float:
        # 同梱の Java 参考実装と同様に、speed 省略時は 1 とする。
        if speed is None:
            return 1.0

        value = self._evaluate_float(speed.expression, parameters)
        # absolute/relative/sequence の意味に従って速度を解決する。
        if speed.type == "absolute":
            return value
        if speed.type == "relative":
            return thread.last_fire_speed + value
        if speed.type == "sequence":
            return thread.last_fire_speed + value
        raise BulletMLRuntimeError(
            f"speed type '{speed.type}' is not supported by the minimal runtime"
        )

    def _move_bullet(self, bullet: BulletState) -> None:
        # 0 度を上、90 度を右とする BulletML の角度系で移動する。
        angle = radians(bullet.direction)
        bullet.x += sin(angle) * bullet.speed + bullet.velocity_x
        bullet.y -= cos(angle) * bullet.speed - bullet.velocity_y

    def _evaluate_float(
        self,
        expression: str,
        parameters: tuple[float, ...] = (),
    ) -> float:
        try:
            return evaluate_expression(
                expression,
                random_func=self._random_func,
                rank=self.rank,
                parameters=parameters,
            )
        except BulletMLExpressionError as exc:
            raise BulletMLRuntimeError(f"failed to evaluate '{expression}'") from exc

    def _evaluate_int(
        self,
        expression: str,
        parameters: tuple[float, ...] = (),
    ) -> int:
        # wait や repeat など、フレーム数として使う式は整数へ切り捨てる。
        return int(self._evaluate_float(expression, parameters))

    def _validate_supported_document(self, document: BulletMLDocument) -> None:
        # パース済み文書に、最小ランタイム未対応の要素が含まれていないか調べる。
        validated_actions: set[int] = set()
        for action in document.actions:
            self._validate_action(action, validated_actions)
        for bullet in document.bullets:
            self._validate_bullet(bullet, validated_actions)
        for fire in document.fires:
            self._validate_fire(fire, validated_actions)

    def _validate_action(
        self,
        action: Action,
        validated_actions: set[int],
    ) -> None:
        # action 内の各命令を再帰的に検証し、未対応命令は実行前に止める。
        action_id = id(action)
        if action_id in validated_actions:
            return
        validated_actions.add(action_id)

        for child in action.children:
            if isinstance(child, Fire):
                self._validate_fire(child, validated_actions)
                continue
            if isinstance(
                child,
                Wait | Vanish | ChangeSpeed | ChangeDirection | Accel,
            ):
                continue
            if isinstance(child, Repeat):
                repeat_action = (
                    child.action
                    if isinstance(child.action, Action)
                    else self._get_referenced_action(child.action)
                )
                self._validate_action(repeat_action, validated_actions)
                continue
            if isinstance(child, Action):
                self._validate_action(child, validated_actions)
                continue
            if isinstance(child, ActionRef):
                self._validate_action(
                    self._get_referenced_action(child),
                    validated_actions,
                )
                continue
            # ここまでの型分岐に該当しない残りの要素は FireRef。
            self._validate_fire(
                self._get_referenced_fire(child),
                validated_actions,
            )

    def _validate_bullet(
        self,
        bullet: Bullet,
        validated_actions: set[int],
    ) -> None:
        for action in bullet.actions:
            resolved_action = (
                action
                if isinstance(action, Action)
                else self._get_referenced_action(action)
            )
            self._validate_action(resolved_action, validated_actions)

    def _validate_fire(
        self,
        fire: Fire,
        validated_actions: set[int],
    ) -> None:
        # 参照解決した bullet の内容も再帰的に検証する。
        bullet = (
            fire.bullet
            if isinstance(fire.bullet, Bullet)
            else self._get_referenced_bullet(fire.bullet)
        )
        self._validate_bullet(bullet, validated_actions)
