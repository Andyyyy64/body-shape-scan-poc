# ローカル実行手順

## この版で動くこと

- CPUで、既知形状・既知変化・尺度誤差・固定形状対照と検出力を検証する。
- 非公開のペア計測データを読み、方式・部位・端末・測定定義ごとの探索的レポートを作る。
- ローカルカメラの録画、動画からのフレーム抽出、チェスボード内部校正、歪み補正。
- 明示的に用意したYOLO segmentation重みから、観測された1人分のマスクを作る。
- **マスクとmetric body-to-camera poseが既にある場合**、透視投影を用いるvisual hullの断面を復元する。
- CUDA環境と公式重みがある場合のSAM単画像初期推定を保存する接続コード。

**自動で回転角・身体位置・実寸を解決する人体スキャナではない。MHRの多視点共同最適化、metric round trip、実人物の精度は未完了。** 現在の`reconstruct`は、推定済み/計測済みのposeを入力する断面方式の比較用ツールで、MHR方式の代用品として合格扱いにはしない。

## インストール

このrepoの実装branchまたは、それを取り込んだcheckoutで実行する。Python 3.11以降を使用する。

```bash
uv sync --locked --extra vision --python 3.11
uv run body-scan doctor
uv run python -m unittest discover -v
```

通常の幾何・統計コマンドはPython標準ライブラリだけでも実行できる。

```bash
python3 -m body_scan --help
python3 -m body_scan power --sigma-m 0.0025 --delta-m 0.01
```

`doctor`の`sam_ready: false`はSAMの環境が揃っていないことを表す。CPUデモの可否とは別。

## まず合成データで動作を確認する

出力先は**どのGit checkoutの中にも置かない**。既存の出力ディレクトリは上書きしないため、再実行は新しいrun名を使う。

```bash
SCAN_DATA_ROOT="$HOME/body-scan-private"
mkdir -p "$SCAN_DATA_ROOT"
umask 077
uv run body-scan demo --out "$SCAN_DATA_ROOT/numeric-01"
uv run body-scan render-demo --out "$SCAN_DATA_ROOT/render-before-01"
uv run body-scan render-demo --delta-m -0.01 --out "$SCAN_DATA_ROOT/render-after-01"
uv run body-scan analyze "$SCAN_DATA_ROOT/numeric-01/pairs.json" \
  --threshold-m 0.003 --target-m 0.01 --out "$SCAN_DATA_ROOT/analysis-01"
```

各runの`report.json`と`report.html`をローカルで読む。`render-demo`は既知pose・既知尺度の合成マスクを作り、実マスクと同じ`reconstruct`へ通す。ラスタ解像度とgridの量子化誤差が残るため、幾何の真値との誤差も表示する。合成結果は身体計測の実証ではない。

## カメラ録画

カメラ起動はこのコマンドを本人が実行した時だけ。OSのカメラ許可が必要。番号は接続カメラに合わせる。

```bash
uv run body-scan capture --source 0 --delay-s 5 --seconds 20 --preview \
  --out "$SCAN_DATA_ROOT/capture-01"
uv run body-scan frames "$SCAN_DATA_ROOT/capture-01/capture.avi" \
  --stride 15 --max-frames 120 --out "$SCAN_DATA_ROOT/frames-01"
```

プレビューではQで停止できる。録画にはフレーム取得時の単調時計の経過秒も保存する。コンテナFPSだけで身体の回転角を推定しない。`frames`は時間的な間引きで、角度選別ではない。`yaw_rad`はnullのまま。

頭・肩・骨盤と、測定したい断面が全方向で映るようにする。腕や下端で断面が欠けたscanを、脚や輪郭の生成補完で救済しない。連続回転と静止回転を別scanにする。

## 校正と座標を揃える

```bash
uv run body-scan board --columns 7 --rows 5 --square-m 0.025 \
  --out "$SCAN_DATA_ROOT/board-01"
```

生成したSVGを100%で印刷し、マス目を実測する。ページに合わせた自動縮小をしない。同じ動画モード・同じ解像度で、板の位置・傾きを十分変えて8枚以上撮る。`columns`/`rows`は内側交点の数。

```bash
uv run body-scan calibrate "$SCAN_DATA_ROOT/board-images" \
  --columns 7 --rows 5 --square-m 0.025 --out "$SCAN_DATA_ROOT/calibration-01"
uv run body-scan undistort "$SCAN_DATA_ROOT/frames-01" \
  --calibration "$SCAN_DATA_ROOT/calibration-01/calibration.json" \
  --out "$SCAN_DATA_ROOT/rectified-01"
```

再投影誤差が小さくても、人体の奥行き・尺度が確定したわけではない。出力は`needs_physical_validation`。`undistort`は画像サイズ・Kを保った全画像変換で、cropやscaleを追加しない。**マスクは補正後の画像から作る**。元画像のマスクを流用しない。

## 観測マスクを作る

```bash
uv sync --locked --extra vision --extra segmentation
uv run body-scan segment "$SCAN_DATA_ROOT/rectified-01" \
  --model "$SCAN_DATA_ROOT/weights/person-seg.pt" \
  --out "$SCAN_DATA_ROOT/masks-01"
```

重みは利用条件を確認したCOCO person segmentationモデルを別途用意する。コマンドは明示されたローカル重みを使い、モデル名からの自動ダウンロードはしない。0人・複数人・座標サイズ不一致を無理に1人へ変換しない。この版のYOLO実推論は未検証で、OpenCVテストの成功と区別する。

手修正マスクも使用できる。マスクはfull-imageサイズのPNGで、0を背景、1または255を人物とする。人のマスク・シルエットも非公開データである。

## 校正済みposeから断面を復元する

入力の完全な例は`render-demo`が作る`scan.json`。実撮影用の空テンプレートも作れる。

```bash
uv run body-scan template --out "$SCAN_DATA_ROOT/input-templates-01"
uv run body-scan reconstruct "$SCAN_DATA_ROOT/render-before-01/scan.json" \
  --out "$SCAN_DATA_ROOT/reconstruction-02"
```

`silhouette_scan.v1`の主要フィールド:

| フィールド | 意味 |
| --- | --- |
| source_kind | real / synthetic。実画像を加工してsyntheticと呼ばない |
| measurement_definition_id | 同じ姿勢・断面位置・曲線定義の版 |
| scale_evidence_id / pose_evidence_id | 尺度と身体poseを拘束した証拠の非公開ID |
| camera.K / distortion / image_size | OpenCVのカメラ座標と元画像サイズ |
| views[].R / t_m | `p_camera = R @ p_body + t_m`。Rは正回転、tはm |
| views[].yaw_rad | 任意の記録用回転角。再構成のcoverageはR/tから得るカメラ位置で再計算 |
| views[].mask | scan.jsonのディレクトリ配下の相対パス |
| bounds_xz_m | 復元平面のxmin/xmax/zmin/zmax |
| sections[].y_m / center_xz_m | 同じbody座標での断面高さと胴体内部の点 |
| grid_step_m / max_yaw_gap_deg | 離散化と観測角度の設定 |

body座標ではyが上、camera座標ではxが右・yが下・zが前。原点は測定定義と一緒に固定する。未知のposeや尺度をテンプレートへ推測で埋めない。尺度の証拠IDがない場合は`unscaled`、整合する断面がない場合は`missing`。画面・復元境界で切れた断面も欠測になる。

得られる周長は胴体seedに連結した領域の**凸包**。皮膚の凹部に沿う周長とは異なる。手足と胴体が連結する高さは人手で測定定義を確認する。離散化誤差は不確かさの実測値ではなく、`uncertainty_m`は未検証の間null。

## SAM初期推定（別のCUDA環境）

公式のSAM環境を構築し、その環境へこのpackageをインストールしてから使う。PyTorchやモデル依存は通常のCPU環境へ自動で追加しない。公式checkpoint、隣接するmodel_config.yaml、MHRモデルを用意する。

```bash
python -m body_scan sam-init "$SCAN_DATA_ROOT/sam-input/scan.json" \
  --source "$SAM_SOURCE" --checkpoint "$SAM_CHECKPOINT" --mhr-model "$MHR_MODEL" \
  --out "$SCAN_DATA_ROOT/sam-initialization-01"
```

各viewに補正済みRGBの`image`と同じ座標の`mask`を指定し、歪み係数はゼロ、Kは補正後のものにする。SAMの初期化にはまだR/tがなくてもよいが、それをmetric再構成が可能という意味にはしない。

完全な公式出力をNPZへ保存し、source commitと重みhashを記録する。出力は`initialized_not_metric_validated`。MHR再生成・単位変換のround tripと共同最適化はこのコマンドに含まれない。実GPUでの推論検証は未実施。信頼できる公式重みだけを使用する。

## ペアの実測比較

`demo`が作る`pairs.json`をschema例として使う。すべてm単位。参照差・参照の不確かさ・予測差・独立したbaseline/followup session ID・subject ID・端末・部位・方式・測定定義・evaluation splitを明示する。失敗attemptは削除せず`failed`/`missing`/`unscaled`とnullを記録する。

しきい値は評価データを見る前に決め、`analyze`へ指定する。参照の不確かさで1cmを下回り得るペアを、確実な1cm変化として感度の分母へ入れない。厳密なnull評価は参照差0・参照不確かさ0のみであり、実人物ではこの条件を満たせない場合がある。その場合は模型でのnull試験と、実人物の誤差モデルを別途用意する。

同じ被験者やbaselineを繰り返し使ったペアは相関ありと表示する。Wilson区間は探索的な記述で、cluster補正済みの確認試験ではない。出力は常に`insufficient_evidence`で、集計が良いだけで製品採用を自動承認しない。相関モデル・オンラインtrendの確率校正は後続の検証。

## 公開前の検査

```bash
uv run body-scan audit --staged --history
```

許可するファイル種類と明らかなパス・credentialの検査を行う。判定結果は匿名化の証明ではないため、追加する文書の実名・身体数値・再識別可能性を人手でも確認する。実データの公開、実行ログの添付、Actions artifactのアップロードはしない。CIは合成データのみで実行する。
