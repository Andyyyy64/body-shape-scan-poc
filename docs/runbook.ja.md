# ローカル実行手順

## 撮影から3D比較まで

1. `serve`で表示されたローカルURLを開く。
2. 「カメラを準備」でMacの内蔵カメラを選ぶ。
3. 撮影時間を設定し、「5秒後に撮影開始」で通常の呼吸のままゆっくり一周する。
4. 保存した撮影の「3Dに復元」を押す。処理は1件ずつ実行する。
5. 完了した撮影をクリックすると、原動画と1回分の3Dを確認できる。
6. 2回目も別に撮影・復元し、基準と比較対象を選んで「差分を計算」を押す。

同じ姿勢・服装・距離で撮り直し、まず日内のばらつきを調べる。日付によるゼロ補正や、前回の体型をそのまま返す処理はない。許容幅は任意の手動設定で、検証済みの生理的変化の閾値ではない。

画面では対応点のモデル上の差と固定断面の周囲長差を表示する。実SAMの指標は、基準で決めた胴体の頂点だけを使う。画面外の脚などはモデルによる補完を含み、測定対象ではない。実寸の独立校正は未完了なので単位は`モデルmm`である。

## Macランタイムを準備する

Hugging Faceの`facebook/sam-3d-body-dinov3`へのアクセス承認とCLIログインを済ませる。トークンをGitやチャットへ保存しない。

```bash
uv sync --locked --extra vision --python 3.11
uv run body-scan setup-mac --out "$HOME/body-scan-private/runtime"
uv run body-scan doctor --runtime-config "$HOME/body-scan-private/runtime/config.json"
uv run body-scan serve --data-root "$HOME/body-scan-private/scans" \
  --runtime-config "$HOME/body-scan-private/runtime/config.json"
```

`setup-mac`は新しいディレクトリ専用。すでにランタイムがある場合は再インストールせず`serve`を使う。モデル約2.8GBと依存を取得し、固定revisionの公式SAM/DINOコードを配置する。GPU・CPUを自動で切り替えず、DINO画像エンコーダーをMPS/float16、デコーダーとMHRをCPUで実行する。必要なdevice指定・ローカルsource指定・mmap読み込みの変更は、期待したソースだけに適用し、不一致なら失敗する。

推論時の外部通信は`macOS sandbox-exec`で遮断する。動画・マスク・meshを外へ送らない。RAM使用量と空きメモリを監視し、上限に達したら元動画を残して停止する。設定値は非公開の`config.json`にある。ほかの重い処理を同時に増やさない。

## 現在の復元方式と検証範囲

動画を実際にデコードしてフレーム数を数え、既定では16枚を選ぶ。ブラウザ録画のシーク情報やFPSだけに依存しない。各フレームでYOLOの人物マスクとSAM 3D Bodyの推定を得て、粗い向きの区間ごとに重みを揃え、shape係数を集約する。

1回目に骨格スケール、正準姿勢の断面位置、胴体の対象頂点を記録し、その後の撮影で共有する。体型のshape係数は各撮影で独立に求める。MHRを同じ姿勢で再生成し、同じ頂点同士を比較する。任意の拡大縮小や非剛体位置合わせで差を消さない。

この方法は`sam_parameter_ensemble`。実際の全方向の輪郭へMHRを共同最適化する方式は、今後の比較対象として残っている。向きが不足した場合は警告し、同じ原動画からの再処理も区別する。比較値を脂肪・筋肉の変化とは断定しない。

実モデルについて確認したのは、公開サンプルでの推論、SAMの完全なMHRパラメータからの頂点再生成、公開画像由来の同じ動画を2回処理する接続試験。本人の独立した一周撮影や、1cmの実変化に対する精度は未検証。

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

`doctor --runtime-config ...`で外部モデル環境を確認する。引数なしの`doctor`は軽量なベース環境だけを診断する。

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

## ペアの実測比較

`demo`が作る`pairs.json`をschema例として使う。すべてm単位。参照差・参照の不確かさ・予測差・独立したbaseline/followup session ID・subject ID・端末・部位・方式・測定定義・evaluation splitを明示する。失敗attemptは削除せず`failed`/`missing`/`unscaled`とnullを記録する。

しきい値は評価データを見る前に決め、`analyze`へ指定する。参照の不確かさで1cmを下回り得るペアを、確実な1cm変化として感度の分母へ入れない。厳密なnull評価は参照差0・参照不確かさ0のみであり、実人物ではこの条件を満たせない場合がある。その場合は模型でのnull試験と、実人物の誤差モデルを別途用意する。

同じ被験者やbaselineを繰り返し使ったペアは相関ありと表示する。Wilson区間は探索的な記述で、cluster補正済みの確認試験ではない。出力は常に`insufficient_evidence`で、集計が良いだけで製品採用を自動承認しない。相関モデル・オンラインtrendの確率校正は後続の検証。

## 公開前の検査

```bash
uv run body-scan audit --staged --history
```

許可するファイル種類と明らかなパス・credentialの検査を行う。判定結果は匿名化の証明ではないため、追加する文書の実名・身体数値・再識別可能性を人手でも確認する。実データの公開、実行ログの添付、Actions artifactのアップロードはしない。CIは合成データのみで実行する。

## 保存済み動画の輪郭フィット（試験）

方向別集約とは別の方法として、各動画内で一つの体型を共有し、
各フレームの人物輪郭へ合わせる処理を追加した。元の動画・meshは変更せず、
同じ保存先に「輪郭フィット（試験）」という新しい撮影レコードを作る。
画面の「一覧を更新」で表示し、この方法の結果同士を比較する。
通常の「3Dに復元」は従来のSAM集約であり、この試験処理は次の明示的なコマンドで実行する。

```sh
/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  "$MODEL_PYTHON" -m body_scan.refine \
  --config "$RUNTIME_CONFIG" --store "$SCAN_STORE" --source "$SOURCE_ID"
```

変数にはセットアップ済みのモデルPython・runtime設定・非公開の保存先・元撮影IDを指定する。
1件ずつ実行し、他のモデル処理とのロック・メモリ上限を守る。結果が不十分でも
元動画と従来結果は残る。`fit.json`には元meshのhash、実装hash、モデルengine、
形状パラメータ、画像上の残差を非公開で保存する。各動画を独立に処理し、
他の撮影の形状や日時・目標周囲長は入力しない。共有する骨格尺度は従来の初回profileを使用する。

これは固定した初期姿勢・フレーム別骨格尺度と輪郭頂点集合を使う局所的な形状最適化であり、
姿勢と形状の完全な共同推定ではない。カメラ並進は最適化するが、焦点距離は元推定値を使用する。
衣服・遮蔽・画面外部分の曖昧さ、初期推定への依存は残る。
輪郭残差が減ったことだけでは実寸精度や再現性を証明しない。
開発に使った動画での1%判定と、未使用の撮影での評価は必ず分ける。
合成データで形状変化への感度・撮影順序への不変性を確認するテストはモデル環境で実行する。

```sh
"$MODEL_PYTHON" -m unittest tests.test_shared_shape -v
```
