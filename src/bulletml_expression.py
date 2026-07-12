from __future__ import annotations

import ast
import random
import re
from collections.abc import Callable, Sequence
from typing import Final


class BulletMLExpressionError(ValueError):
    # BulletML の数式として解釈できない入力を呼び出し側へ伝える。
    pass


# Python の AST ノードを、BulletML の最小実装で許可する二項演算へ対応付ける。
_BINARY_OPERATORS: Final[dict[type[ast.operator], Callable[[float, float], float]]] = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Mod: lambda left, right: left % right,
}


# 単項の符号演算だけを許可し、関数呼び出しなどは評価対象にしない。
_UNARY_OPERATORS: Final[dict[type[ast.unaryop], Callable[[float], float]]] = {
    ast.UAdd: lambda operand: operand,
    ast.USub: lambda operand: -operand,
}


# BulletML 固有変数は Python の識別子として無効なので、
# AST パース前に内部識別子へ置換する。
_RAND_IDENTIFIER_PREFIX: Final = "__bulletml_rand_"
_RANK_IDENTIFIER: Final = "__bulletml_rank"
_PARAMETER_IDENTIFIER_PREFIX: Final = "__bulletml_parameter_"

# $rand 単独トークンだけを対象にする。
# 例:
#   "$rand"          -> 対象
#   "360 * $rand"    -> 対象
#   "$random"        -> 対象外
#   "foo$rand"       -> 対象外
_RAND_PATTERN: Final = re.compile(r"(?<![A-Za-z0-9_$])\$rand(?![A-Za-z0-9_])")
_RANK_PATTERN: Final = re.compile(r"(?<![A-Za-z0-9_$])\$rank(?![A-Za-z0-9_])")
_PARAMETER_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_$])\$([0-9]+)(?![A-Za-z0-9_])"
)


def evaluate_expression(
    expression: str,
    *,
    random_func: Callable[[], float] = random.random,
    rank: float = 0.5,
    parameters: Sequence[float] = (),
) -> float:
    # 内部識別子を利用者が直接記述することは禁止する。
    for reserved_identifier in (
        _RAND_IDENTIFIER_PREFIX,
        _RANK_IDENTIFIER,
        _PARAMETER_IDENTIFIER_PREFIX,
    ):
        if reserved_identifier in expression:
            raise BulletMLExpressionError(
                f"reserved identifier is not allowed: {reserved_identifier}"
            )

    # 各 $rand を別々の内部識別子へ置換し、出現ごとに乱数を生成できるようにする。
    rand_index = 0

    def replace_rand(_match: re.Match[str]) -> str:
        nonlocal rand_index
        identifier = f"{_RAND_IDENTIFIER_PREFIX}{rand_index}"
        rand_index += 1
        return identifier

    normalized_expression = _RAND_PATTERN.sub(replace_rand, expression)
    normalized_expression = _RANK_PATTERN.sub(
        _RANK_IDENTIFIER,
        normalized_expression,
    )
    normalized_expression = _PARAMETER_PATTERN.sub(
        _replace_parameter,
        normalized_expression,
    )

    # 文字列を eval せず AST として読み、
    # 許可したノードだけを後段で評価する。
    try:
        parsed = ast.parse(
            normalized_expression,
            mode="eval",
        )
    except SyntaxError as exc:
        raise BulletMLExpressionError(f"invalid expression: {expression}") from exc

    # 実行時エラーは BulletML 用の例外へ包み直して、
    # ランタイム側の扱いを揃える。
    try:
        return float(
            _evaluate_node(
                parsed.body,
                random_func=random_func,
                rank_value=float(rank),
                parameter_values=tuple(float(value) for value in parameters),
            )
        )
    except ZeroDivisionError as exc:
        raise BulletMLExpressionError("division by zero") from exc


def _evaluate_node(
    node: ast.AST,
    *,
    random_func: Callable[[], float],
    rank_value: float,
    parameter_values: tuple[float, ...],
) -> float:
    # 数値リテラルはそのまま float 化して計算の基準値にする。
    if isinstance(node, ast.Constant):
        value = node.value

        # bool は int の派生型なので明示的に除外する。
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    # $rand は Java 参考実装と同様に、式中の出現ごとに乱数を生成する。
    if isinstance(node, ast.Name) and node.id.startswith(_RAND_IDENTIFIER_PREFIX):
        return float(random_func())

    # $rank はランタイムから渡された現在の難易度値を返す。
    if isinstance(node, ast.Name) and node.id == _RANK_IDENTIFIER:
        return rank_value

    # $1, $2, ... は呼び出し側から渡された値を 1 始まりで参照する。
    if isinstance(node, ast.Name) and node.id.startswith(_PARAMETER_IDENTIFIER_PREFIX):
        parameter_number = int(node.id.removeprefix(_PARAMETER_IDENTIFIER_PREFIX))
        parameter_index = parameter_number - 1
        if parameter_index >= len(parameter_values):
            raise BulletMLExpressionError(
                f"positional parameter ${parameter_number} is unavailable; "
                f"received {len(parameter_values)} parameter(s)"
            )
        return parameter_values[parameter_index]

    # 単項演算はオペランドを再帰的に評価してから適用する。
    if isinstance(node, ast.UnaryOp):
        operator = _UNARY_OPERATORS.get(type(node.op))

        if operator is not None:
            return operator(
                _evaluate_node(
                    node.operand,
                    random_func=random_func,
                    rank_value=rank_value,
                    parameter_values=parameter_values,
                )
            )

    # 二項演算は左右の式を再帰的に評価し、
    # 許可済み演算子だけを実行する。
    if isinstance(node, ast.BinOp):
        operator = _BINARY_OPERATORS.get(type(node.op))

        if operator is not None:
            return operator(
                _evaluate_node(
                    node.left,
                    random_func=random_func,
                    rank_value=rank_value,
                    parameter_values=parameter_values,
                ),
                _evaluate_node(
                    node.right,
                    random_func=random_func,
                    rank_value=rank_value,
                    parameter_values=parameter_values,
                ),
            )

    # その他の名前参照、関数呼び出し、属性アクセスなどは拒否する。
    raise BulletMLExpressionError(f"unsupported expression: {ast.dump(node)}")


def _replace_parameter(match: re.Match[str]) -> str:
    parameter_number = int(match.group(1))
    if parameter_number < 1:
        raise BulletMLExpressionError(
            "positional parameter index must be 1 or greater: $0"
        )
    return f"{_PARAMETER_IDENTIFIER_PREFIX}{parameter_number}"
