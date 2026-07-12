from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypeVar, cast

# BulletML の属性値を Python 側の型として表現する。
DocumentType = Literal["none", "vertical", "horizontal"]
DirectionType = Literal["aim", "absolute", "relative", "sequence"]
VectorType = Literal["absolute", "relative", "sequence"]

# XML から読んだ属性値を検証するための許可リスト。
_DOCUMENT_TYPES: Final[set[str]] = {"none", "vertical", "horizontal"}
_DIRECTION_TYPES: Final[set[str]] = {"aim", "absolute", "relative", "sequence"}
_VECTOR_TYPES: Final[set[str]] = {"absolute", "relative", "sequence"}


class BulletMLParseError(ValueError):
    def __init__(self, message: str, *, element: str | None = None) -> None:
        # エラーが起きた要素名を付けて、XML の修正箇所を追いやすくする。
        self.element = element
        if element is None:
            super().__init__(message)
            return
        super().__init__(f"{element}: {message}")


# XML の値要素は、式文字列と type 属性をそのまま保持する。
@dataclass(frozen=True, slots=True)
class Direction:
    expression: str
    type: DirectionType = "aim"


@dataclass(frozen=True, slots=True)
class Speed:
    expression: str
    type: VectorType = "absolute"


@dataclass(frozen=True, slots=True)
class Horizontal:
    expression: str
    type: VectorType = "absolute"


@dataclass(frozen=True, slots=True)
class Vertical:
    expression: str
    type: VectorType = "absolute"


@dataclass(frozen=True, slots=True)
class Times:
    expression: str


@dataclass(frozen=True, slots=True)
class Term:
    expression: str


@dataclass(frozen=True, slots=True)
class Param:
    expression: str


@dataclass(frozen=True, slots=True)
class Wait:
    expression: str


# vanish は空要素として扱うため、追加の値を持たない。
@dataclass(frozen=True, slots=True)
class Vanish:
    pass


# Ref 系要素は label と param 群だけを保持し、解決はランタイム側に委ねる。
@dataclass(frozen=True, slots=True)
class ActionRef:
    label: str
    params: tuple[Param, ...] = ()


@dataclass(frozen=True, slots=True)
class BulletRef:
    label: str
    params: tuple[Param, ...] = ()


@dataclass(frozen=True, slots=True)
class FireRef:
    label: str
    params: tuple[Param, ...] = ()


# 制御命令と弾の変化命令は、構文順を検証したあとデータとして保持する。
@dataclass(frozen=True, slots=True)
class Repeat:
    times: Times
    action: ActionOrRef


@dataclass(frozen=True, slots=True)
class ChangeSpeed:
    speed: Speed
    term: Term


@dataclass(frozen=True, slots=True)
class ChangeDirection:
    direction: Direction
    term: Term


@dataclass(frozen=True, slots=True)
class Accel:
    horizontal: Horizontal | None
    vertical: Vertical | None
    term: Term


# Action/Bullet/Fire は BulletML の主要ノードを構文木として表す。
@dataclass(frozen=True, slots=True)
class Action:
    label: str | None = None
    children: tuple[ActionChild, ...] = ()


@dataclass(frozen=True, slots=True)
class Bullet:
    label: str | None = None
    direction: Direction | None = None
    speed: Speed | None = None
    actions: tuple[ActionOrRef, ...] = ()


@dataclass(frozen=True, slots=True)
class Fire:
    bullet: BulletOrRef
    label: str | None = None
    direction: Direction | None = None
    speed: Speed | None = None


# 再帰的な BulletML 構文を型エイリアスで表し、パース結果の形を明確にする。
ActionOrRef: TypeAlias = Action | ActionRef
BulletOrRef: TypeAlias = Bullet | BulletRef
ActionChild: TypeAlias = (
    Repeat
    | Fire
    | FireRef
    | ChangeSpeed
    | ChangeDirection
    | Accel
    | Wait
    | Vanish
    | Action
    | ActionRef
)
TopLevelNode: TypeAlias = Action | Bullet | Fire


@dataclass(frozen=True, slots=True)
class BulletMLDocument:
    contents: tuple[TopLevelNode, ...] = ()
    type: DocumentType = "none"
    xmlns: str | None = None

    @property
    def actions(self) -> tuple[Action, ...]:
        # トップレベルの action だけを取り出し、ラベル解決用の入力にする。
        return tuple(node for node in self.contents if isinstance(node, Action))

    @property
    def bullets(self) -> tuple[Bullet, ...]:
        # トップレベルの bullet だけを取り出す。
        return tuple(node for node in self.contents if isinstance(node, Bullet))

    @property
    def fires(self) -> tuple[Fire, ...]:
        # トップレベルの fire だけを取り出す。
        return tuple(node for node in self.contents if isinstance(node, Fire))


# ラベルインデックスを共通化するため、label を持つトップレベルノードを表す。
_LabeledNode = TypeVar("_LabeledNode", Action, Bullet, Fire)


def load_bulletml(path: str | Path) -> BulletMLDocument:
    # ファイルから XML を読み込み、ElementTree のエラーを BulletML 用に整える。
    xml_path = Path(path)
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise BulletMLParseError(_format_xml_parse_error(exc)) from exc
    return _parse_document(root)


def parse_bulletml(xml_text: str) -> BulletMLDocument:
    # テストや文字列入力向けに、XML テキストから直接ドキュメントを作る。
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise BulletMLParseError(_format_xml_parse_error(exc)) from exc
    return _parse_document(root)


def action_index(document: BulletMLDocument) -> dict[str, Action]:
    # top-level action を label で引ける辞書にする。
    return _build_label_index(document.actions, "action")


def bullet_index(document: BulletMLDocument) -> dict[str, Bullet]:
    # top-level bullet を label で引ける辞書にする。
    return _build_label_index(document.bullets, "bullet")


def fire_index(document: BulletMLDocument) -> dict[str, Fire]:
    # top-level fire を label で引ける辞書にする。
    return _build_label_index(document.fires, "fire")


def _build_label_index(
    nodes: tuple[_LabeledNode, ...], kind: str
) -> dict[str, _LabeledNode]:
    # label のないノードは参照対象にせず、重複 label は早い段階で検出する。
    index: dict[str, _LabeledNode] = {}
    for node in nodes:
        if node.label is None:
            continue
        if node.label in index:
            raise ValueError(f"duplicate {kind} label: {node.label}")
        index[node.label] = node
    return index


def _parse_document(element: ET.Element) -> BulletMLDocument:
    # ルート要素が bulletml であることと、許可属性だけを持つことを確認する。
    xmlns, name = _split_tag(element.tag)
    if name != "bulletml":
        raise BulletMLParseError("root element must be bulletml", element=name)
    _ensure_allowed_attributes(element, {"type"})
    _ensure_whitespace_only_container(element, "bulletml")

    document_type = cast(
        DocumentType,
        _parse_enum(
            element.attrib.get("type", "none"), _DOCUMENT_TYPES, "type", "bulletml"
        ),
    )

    # top-level では action / bullet / fire のみを順番に構文木へ変換する。
    contents: list[TopLevelNode] = []
    for child in _child_elements(element):
        child_name = _local_name(child)
        if child_name == "action":
            contents.append(_parse_action(child))
            continue
        if child_name == "bullet":
            contents.append(_parse_bullet(child))
            continue
        if child_name == "fire":
            contents.append(_parse_fire(child))
            continue
        raise BulletMLParseError(f"unexpected child <{child_name}>", element="bulletml")
    return BulletMLDocument(contents=tuple(contents), type=document_type, xmlns=xmlns)


def _parse_action(element: ET.Element) -> Action:
    # action は制御命令を並べた実行単位なので、子要素を命令として順に読む。
    _ensure_allowed_attributes(element, {"label"})
    _ensure_whitespace_only_container(element, "action")
    children: list[ActionChild] = []
    for child in _child_elements(element):
        child_name = _local_name(child)
        if child_name == "repeat":
            children.append(_parse_repeat(child))
            continue
        if child_name == "fire":
            children.append(_parse_fire(child))
            continue
        if child_name == "fireRef":
            children.append(_parse_fire_ref(child))
            continue
        if child_name == "changeSpeed":
            children.append(_parse_change_speed(child))
            continue
        if child_name == "changeDirection":
            children.append(_parse_change_direction(child))
            continue
        if child_name == "accel":
            children.append(_parse_accel(child))
            continue
        if child_name == "wait":
            children.append(_parse_wait(child))
            continue
        if child_name == "vanish":
            children.append(_parse_vanish(child))
            continue
        if child_name == "action":
            children.append(_parse_action(child))
            continue
        if child_name == "actionRef":
            children.append(_parse_action_ref(child))
            continue
        raise BulletMLParseError(f"unexpected child <{child_name}>", element="action")
    return Action(label=element.attrib.get("label"), children=tuple(children))


def _parse_bullet(element: ET.Element) -> Bullet:
    # bullet は direction?, speed?, action/actionRef* の順序だけを許可する。
    _ensure_allowed_attributes(element, {"label"})
    _ensure_whitespace_only_container(element, "bullet")
    children = _child_elements(element)
    index = 0

    direction: Direction | None = None
    if index < len(children) and _local_name(children[index]) == "direction":
        direction = _parse_direction(children[index])
        index += 1

    speed: Speed | None = None
    if index < len(children) and _local_name(children[index]) == "speed":
        speed = _parse_speed(children[index])
        index += 1

    actions: list[ActionOrRef] = []
    while index < len(children):
        # direction/speed の後ろに残る要素は、弾自身の action 群として扱う。
        child = children[index]
        child_name = _local_name(child)
        if child_name == "action":
            actions.append(_parse_action(child))
        elif child_name == "actionRef":
            actions.append(_parse_action_ref(child))
        else:
            raise BulletMLParseError(
                f"unexpected child <{child_name}> after direction/speed section",
                element="bullet",
            )
        index += 1

    return Bullet(
        label=element.attrib.get("label"),
        direction=direction,
        speed=speed,
        actions=tuple(actions),
    )


def _parse_fire(element: ET.Element) -> Fire:
    # fire は direction?, speed?, bullet/bulletRef の順で、弾定義は 1 つだけ許可する。
    _ensure_allowed_attributes(element, {"label"})
    _ensure_whitespace_only_container(element, "fire")
    children = _child_elements(element)
    index = 0

    direction: Direction | None = None
    if index < len(children) and _local_name(children[index]) == "direction":
        direction = _parse_direction(children[index])
        index += 1

    speed: Speed | None = None
    if index < len(children) and _local_name(children[index]) == "speed":
        speed = _parse_speed(children[index])
        index += 1

    if index >= len(children):
        raise BulletMLParseError(
            "fire requires a bullet or bulletRef child", element="fire"
        )

    bullet_element = children[index]
    bullet_name = _local_name(bullet_element)
    # inline bullet と bulletRef のどちらかを読み、余分な子要素は後で拒否する。
    if bullet_name == "bullet":
        bullet: BulletOrRef = _parse_bullet(bullet_element)
    elif bullet_name == "bulletRef":
        bullet = _parse_bullet_ref(bullet_element)
    else:
        raise BulletMLParseError(
            f"expected <bullet> or <bulletRef>, got <{bullet_name}>",
            element="fire",
        )
    index += 1

    if index != len(children):
        raise BulletMLParseError(
            "fire must contain exactly one bullet child", element="fire"
        )

    return Fire(
        bullet=bullet,
        label=element.attrib.get("label"),
        direction=direction,
        speed=speed,
    )


def _parse_action_ref(element: ET.Element) -> ActionRef:
    # 参照先 label と呼び出し時の param 群を保持する。
    return ActionRef(
        label=_required_label(element, "actionRef"),
        params=_parse_params(element, "actionRef"),
    )


def _parse_bullet_ref(element: ET.Element) -> BulletRef:
    # 参照先 bullet の label と param 群を保持する。
    return BulletRef(
        label=_required_label(element, "bulletRef"),
        params=_parse_params(element, "bulletRef"),
    )


def _parse_fire_ref(element: ET.Element) -> FireRef:
    # 参照先 fire の label と param 群を保持する。
    return FireRef(
        label=_required_label(element, "fireRef"),
        params=_parse_params(element, "fireRef"),
    )


def _parse_repeat(element: ET.Element) -> Repeat:
    # repeat は times の直後に action/actionRef が 1 つ続く形だけを許可する。
    _ensure_allowed_attributes(element, set())
    _ensure_whitespace_only_container(element, "repeat")
    children = _child_elements(element)
    if len(children) != 2:
        raise BulletMLParseError(
            "repeat requires <times> followed by <action> or <actionRef>",
            element="repeat",
        )

    if _local_name(children[0]) != "times":
        raise BulletMLParseError("first child must be <times>", element="repeat")
    times = _parse_times(children[0])

    action_child = children[1]
    action_name = _local_name(action_child)
    if action_name == "action":
        action: ActionOrRef = _parse_action(action_child)
    elif action_name == "actionRef":
        action = _parse_action_ref(action_child)
    else:
        raise BulletMLParseError(
            "second child must be <action> or <actionRef>", element="repeat"
        )
    return Repeat(times=times, action=action)


def _parse_change_speed(element: ET.Element) -> ChangeSpeed:
    # changeSpeed は speed と term をこの順序で持つ必要がある。
    _ensure_allowed_attributes(element, set())
    _ensure_whitespace_only_container(element, "changeSpeed")
    children = _child_elements(element)
    if len(children) != 2:
        raise BulletMLParseError(
            "changeSpeed requires <speed> and <term>", element="changeSpeed"
        )
    if _local_name(children[0]) != "speed" or _local_name(children[1]) != "term":
        raise BulletMLParseError(
            "children must be <speed> followed by <term>", element="changeSpeed"
        )
    return ChangeSpeed(speed=_parse_speed(children[0]), term=_parse_term(children[1]))


def _parse_change_direction(element: ET.Element) -> ChangeDirection:
    # changeDirection は direction と term をこの順序で持つ必要がある。
    _ensure_allowed_attributes(element, set())
    _ensure_whitespace_only_container(element, "changeDirection")
    children = _child_elements(element)
    if len(children) != 2:
        raise BulletMLParseError(
            "changeDirection requires <direction> and <term>",
            element="changeDirection",
        )
    if _local_name(children[0]) != "direction" or _local_name(children[1]) != "term":
        raise BulletMLParseError(
            "children must be <direction> followed by <term>",
            element="changeDirection",
        )
    return ChangeDirection(
        direction=_parse_direction(children[0]), term=_parse_term(children[1])
    )


def _parse_accel(element: ET.Element) -> Accel:
    # accel は horizontal?, vertical?, term の順で速度成分の変化を表す。
    _ensure_allowed_attributes(element, set())
    _ensure_whitespace_only_container(element, "accel")
    children = _child_elements(element)
    index = 0

    horizontal: Horizontal | None = None
    if index < len(children) and _local_name(children[index]) == "horizontal":
        horizontal = _parse_horizontal(children[index])
        index += 1

    vertical: Vertical | None = None
    if index < len(children) and _local_name(children[index]) == "vertical":
        vertical = _parse_vertical(children[index])
        index += 1

    if index >= len(children) or _local_name(children[index]) != "term":
        raise BulletMLParseError("accel requires a trailing <term>", element="accel")
    # 最後の term を必須にして、加速度の補間フレーム数を保持する。
    term = _parse_term(children[index])
    index += 1

    if index != len(children):
        raise BulletMLParseError(
            "accel only allows <horizontal?> <vertical?> <term>", element="accel"
        )
    return Accel(horizontal=horizontal, vertical=vertical, term=term)


def _parse_direction(element: ET.Element) -> Direction:
    # direction の type 省略時は BulletML の既定値 aim として読む。
    _ensure_allowed_attributes(element, {"type"})
    direction_type = cast(
        DirectionType,
        _parse_enum(
            element.attrib.get("type", "aim"), _DIRECTION_TYPES, "type", "direction"
        ),
    )
    return Direction(
        expression=_parse_text_node(element, "direction"), type=direction_type
    )


def _parse_speed(element: ET.Element) -> Speed:
    # speed の type 省略時は absolute として読む。
    _ensure_allowed_attributes(element, {"type"})
    speed_type = cast(
        VectorType,
        _parse_enum(
            element.attrib.get("type", "absolute"), _VECTOR_TYPES, "type", "speed"
        ),
    )
    return Speed(expression=_parse_text_node(element, "speed"), type=speed_type)


def _parse_horizontal(element: ET.Element) -> Horizontal:
    # accel 用の水平成分を、式文字列と type で保持する。
    _ensure_allowed_attributes(element, {"type"})
    horizontal_type = cast(
        VectorType,
        _parse_enum(
            element.attrib.get("type", "absolute"), _VECTOR_TYPES, "type", "horizontal"
        ),
    )
    return Horizontal(
        expression=_parse_text_node(element, "horizontal"), type=horizontal_type
    )


def _parse_vertical(element: ET.Element) -> Vertical:
    # accel 用の垂直成分を、式文字列と type で保持する。
    _ensure_allowed_attributes(element, {"type"})
    vertical_type = cast(
        VectorType,
        _parse_enum(
            element.attrib.get("type", "absolute"), _VECTOR_TYPES, "type", "vertical"
        ),
    )
    return Vertical(
        expression=_parse_text_node(element, "vertical"), type=vertical_type
    )


def _parse_times(element: ET.Element) -> Times:
    # repeat の回数式をテキスト要素として読む。
    _ensure_allowed_attributes(element, set())
    return Times(expression=_parse_text_node(element, "times"))


def _parse_term(element: ET.Element) -> Term:
    # 変化にかけるフレーム数の式を読む。
    _ensure_allowed_attributes(element, set())
    return Term(expression=_parse_text_node(element, "term"))


def _parse_param(element: ET.Element) -> Param:
    # Ref 呼び出し時に渡す引数式を読む。
    _ensure_allowed_attributes(element, set())
    return Param(expression=_parse_text_node(element, "param"))


def _parse_wait(element: ET.Element) -> Wait:
    # action 内の待機フレーム数を表す式を読む。
    _ensure_allowed_attributes(element, set())
    return Wait(expression=_parse_text_node(element, "wait"))


def _parse_vanish(element: ET.Element) -> Vanish:
    # vanish は空要素として扱い、子要素や本文があれば構文エラーにする。
    _ensure_allowed_attributes(element, set())
    if list(element):
        raise BulletMLParseError("vanish cannot have child elements", element="vanish")
    if _normalized_text(element.text):
        raise BulletMLParseError("vanish cannot contain text", element="vanish")
    return Vanish()


def _parse_params(element: ET.Element, element_name: str) -> tuple[Param, ...]:
    # Ref 系要素では label 属性以外に、param 子要素だけを並べられる。
    _ensure_allowed_attributes(element, {"label"})
    _ensure_whitespace_only_container(element, element_name)
    params: list[Param] = []
    for child in _child_elements(element):
        child_name = _local_name(child)
        if child_name != "param":
            raise BulletMLParseError(
                f"unexpected child <{child_name}>", element=element_name
            )
        params.append(_parse_param(child))
    return tuple(params)


def _required_label(element: ET.Element, element_name: str) -> str:
    # 参照要素に必要な label 属性を取り出し、空文字は拒否する。
    label = element.attrib.get("label")
    if label is None or not label.strip():
        raise BulletMLParseError("label attribute is required", element=element_name)
    return label


def _parse_text_node(element: ET.Element, element_name: str) -> str:
    # 値要素はテキストだけを許可し、子要素や混在テキストを拒否する。
    if list(element):
        raise BulletMLParseError(
            "text-only element cannot contain child elements", element=element_name
        )
    _ensure_no_significant_tail_children(element, element_name)
    return _normalized_text(element.text)


def _ensure_allowed_attributes(element: ET.Element, allowed: set[str]) -> None:
    # 未知の属性を検出し、最小実装が暗黙に無視しないようにする。
    element_name = _local_name(element)
    unexpected = sorted(attr for attr in element.attrib if attr not in allowed)
    if unexpected:
        joined = ", ".join(unexpected)
        raise BulletMLParseError(
            f"unexpected attribute(s): {joined}", element=element_name
        )


def _ensure_whitespace_only_container(element: ET.Element, element_name: str) -> None:
    # コンテナ要素は子要素の並びだけを意味として持ち、本文テキストは許可しない。
    if _normalized_text(element.text):
        raise BulletMLParseError("unexpected text content", element=element_name)
    _ensure_no_significant_tail_children(element, element_name)


def _ensure_no_significant_tail_children(
    element: ET.Element, element_name: str
) -> None:
    # 子要素の直後にある tail テキストも、空白以外なら混在テキストとして拒否する。
    for child in element:
        if _normalized_text(child.tail):
            raise BulletMLParseError(
                "mixed text content is not supported", element=element_name
            )


def _parse_enum(
    value: str, allowed: set[str], attribute_name: str, element_name: str
) -> str:
    # type などの列挙属性が、仕様で許可された値だけを取るように検証する。
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise BulletMLParseError(
            f"invalid {attribute_name}={value!r}; expected one of: {options}",
            element=element_name,
        )
    return value


def _child_elements(element: ET.Element) -> list[ET.Element]:
    # ElementTree の子要素イテレータを、添字で順序検査できる list にする。
    return list(element)


def _local_name(element: ET.Element) -> str:
    # 名前空間が付いていても、構文判定ではローカル名だけを使う。
    return _split_tag(element.tag)[1]


def _split_tag(tag: str) -> tuple[str | None, str]:
    # ElementTree の {namespace}local 形式を namespace と local name に分ける。
    if tag.startswith("{"):
        namespace, local_name = tag[1:].split("}", 1)
        return namespace, local_name
    return None, tag


def _normalized_text(text: str | None) -> str:
    # 空白だけのテキストは構文上の意味を持たないものとして扱う。
    if text is None:
        return ""
    return text.strip()


def _format_xml_parse_error(error: ET.ParseError) -> str:
    # ElementTree の位置情報を、利用者向けのエラーメッセージに含める。
    line, column = error.position
    return f"invalid XML at line {line}, column {column}: {error}"


# 外部から直接使うパーサ型と関数だけを公開する。
__all__ = [
    "Accel",
    "Action",
    "ActionRef",
    "Bullet",
    "BulletMLDocument",
    "BulletMLParseError",
    "BulletRef",
    "ChangeDirection",
    "ChangeSpeed",
    "Direction",
    "Fire",
    "FireRef",
    "Horizontal",
    "Param",
    "Repeat",
    "Speed",
    "Term",
    "Times",
    "Vanish",
    "Vertical",
    "Wait",
    "action_index",
    "bullet_index",
    "fire_index",
    "load_bulletml",
    "parse_bulletml",
]
