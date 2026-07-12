# bulletml_runtime

[BulletML](https://www.asahi-net.or.jp/~cs8k-cyu/bulletml/) の XML を Python で解釈して弾幕を実行するためのライブラリ。

## インストール
src以下のファイルをコピーして使ってください。

## Parser の使い方

ファイルから読み込む場合は `load_bulletml()` を使います。

```python
from src.bulletml_parser import (
    BulletMLParseError,
    action_index,
    bullet_index,
    fire_index,
    load_bulletml,
)

try:
    document = load_bulletml("simple_barrage.xml")
except (BulletMLParseError, OSError) as exc:
    print(f"BulletML の読み込みに失敗しました: {exc}")
    raise

print(document.type)       # "none", "vertical", "horizontal" のいずれか
print(document.xmlns)      # XML 名前空間。指定がなければ None
print(document.actions)    # トップレベルの Action
print(document.bullets)    # トップレベルの Bullet
print(document.fires)      # トップレベルの Fire

# label をキーにして定義を参照できます。
actions = action_index(document)
bullets = bullet_index(document)
fires = fire_index(document)
print(actions["huntingFan"])
```

XML 文字列を直接パースする場合は `parse_bulletml()` を使います。

```python
from src.bulletml_parser import parse_bulletml

xml_text = """
<bulletml>
  <action label="top">
    <wait>60</wait>
  </action>
</bulletml>
"""

document = parse_bulletml(xml_text)
```

## Runtime の使い方

パースしたドキュメント、発射元の座標、狙い先の現在座標を返す関数を渡して `BulletMLRuntime` を作成します。その後、ゲームループからフレームごとに `step()` を呼び出します。

```python
from src.bulletml_parser import load_bulletml
from src.bulletml_runtime import BulletMLRuntime, BulletMLRuntimeError

document = load_bulletml("simple_barrage.xml")

# プレイヤーが移動しても最新の座標を返せるよう、関数として渡します。
player_position = [160.0, 220.0]

try:
    runtime = BulletMLRuntime(
        document,
        origin_x=160.0,
        origin_y=48.0,
        target_position=lambda: (player_position[0], player_position[1]),
        rank=0.5,
    )
except BulletMLRuntimeError as exc:
    print(f"BulletML を実行できません: {exc}")
    raise

def update() -> None:
    # 1フレーム分、発射処理・弾の action・移動を進めます。
    runtime.step()

    # 画面外の弾はホストアプリケーション側で除去します。
    runtime.remove_bullets_if(
        lambda bullet: (
            bullet.x < 0
            or bullet.x >= 320
            or bullet.y < 0
            or bullet.y >= 240
        )
    )

    for bullet in runtime.bullets:
        # 利用する描画ライブラリへ現在位置を渡します。
        print(bullet.x, bullet.y)
```

`runtime.bullets` は現在生存している弾の `tuple` です。各弾から主に次の状態を参照できます。

- `x`, `y`: 現在位置
- `direction`: 進行方向（度）。`0` が上、`90` が右です
- `speed`: 進行方向に対する速度
- `velocity_x`, `velocity_y`: `<accel>` による各軸の速度
- `alive`: 生存状態

### 起点となる action

既定では、label が `top` で始まるトップレベル action（例: `top`, `top1`, `topAttack`）をすべて並行実行します。別の action だけを起点にする場合は、その label を `root_action_label` に指定します。

```python
runtime = BulletMLRuntime(
    document,
    origin_x=160.0,
    origin_y=48.0,
    target_position=lambda: (160.0, 220.0),
    root_action_label="huntingFan",
)
```

`top` で始まる action がなく、トップレベル action が1つだけの場合は、その action が自動的に起点になります。起点を特定できない場合は `BulletMLRuntimeError` が送出されます。

### 式と難易度

BulletML の数式では、四則演算、剰余、単項の `+` / `-` に加え、`$rand`、`$rank`、`$1` 以降の位置引数を利用できます。`$rank` の値はコンストラクタの `rank` で指定します。再現可能な乱数が必要な場合は、`random_func` に値を返す関数を渡してください。

```python
import random

rng = random.Random(1234)
runtime = BulletMLRuntime(
    document,
    origin_x=160.0,
    origin_y=48.0,
    target_position=lambda: (160.0, 220.0),
    rank=0.8,
    random_func=rng.random,
)
```

## サンプルの実行

Pyxel を使ったサンプルビューアーを実行できます。

```console
uv run python viewer.py
```

## License

[0BSD](LICENSE)
