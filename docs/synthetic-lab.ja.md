# 合成動画による自動検証

実撮影なしで、人物segmentation → SAM 3D Body → per-frame QC → 輪郭fit →
canonical MHR meshの経路を検証する。通常の実験では正解mesh・camera・輪郭は推定器へ渡さず、
入力はレンダリングされた動画だけにする。下記のカメラ比較だけは宣言済みの内部パラメータを
明示的に渡す。正解形状・変形量はどの推定にも渡さない。実人体の精度、実測cmへの校正、
Somaの採用ゲートはこの実験だけで満たされない。

```bash
uv run python -m body_scan.synthetic_lab init \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json" \
  --out "$HOME/body-scan-private/synthetic-lab-new"
uv run python -m body_scan.synthetic_lab run \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json" \
  --out "$HOME/body-scan-private/synthetic-lab-new"
# 中断後は同じrunコマンドで未完了ケースから続ける。
# terminalの失敗は自動的に消去・再試行しない。
# --cases baseline_17 stress_blank のように名前を指定できる。
# initにも--casesを付ければ、初めから指定したケースだけを計画する。
uv run python -m body_scan.synthetic_evidence \
  --out "$HOME/body-scan-private/synthetic-lab-new"
uv run python -m body_scan.synthetic_diagnostics \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json" \
  --out "$HOME/body-scan-private/synthetic-lab-new"
# diagnostics後にsynthetic_evidenceを再実行すると、初期推定/fitの表も追加される。
uv run python -m body_scan.synthetic_projection \
  --out "$HOME/body-scan-private/synthetic-lab-new"
# 保存された推定とgeneratorの肩/腰の画像座標・相対yaw・変形pixel量を事後比較する。
# この正解情報は推定器へ渡さず、誤差の大きさの比較を因果の証明とは扱わない。
```

微小変化の比較用に、連続投影座標と8倍の画素面積近似を使うrendererを選べる。
新しい実験の`init`に`--renderer continuous_coverage8.v1`を付けると、
各caseのspecとprotocolに選択が保存される。出力解像度は640×640のままで、
光・遮蔽・blurなどの悪条件は従来と同じ処理を通る。
指定を省略した実験は従来の`legacy_integer.v1`を使う。
`run`は保存済みの選択を使うため、実行途中でrendererを変更できない。
既存の動画・実験を差し替えず、別の出力ディレクトリで比較する。
rendererの画素面積の忠実度と、segmentation・測定算法・実人体での精度は別々に検証する。

モデルのexportと推論は既存の通信隔離・モデルchecksum・共有inference lock・
メモリ制限を使う。実撮影のstoreは使わない。合成storeのmarkerとworkerの
`--allow-synthetic`の両方が必要で、出力の`source_kind`は`synthetic`を維持する。
実撮影と合成のprofile/比較は共有できない。通常の撮影UIから合成workerは
起動しない。

33ケースを順次処理する。基準3回、局所腹囲変形5条件（−2/−1/+1/+2/+4cm相当）、
距離1.5/2/2.5モデルm各2回、撮影条件19種類。条件には背景、暗さ、逆光、
ぼけ、noise、JPEG劣化、camera揺れ、呼吸様の変形、衣服様の表面、左右非対称、
大小の遮蔽、複数人物、頭の切れ、回転不足、無人、明滅、複合劣化を含める。
寸法は合成座標上の`model_m`で、人体の実測メートルではない。

各ケースに動画、preview、truth、入力hash、全フレームのQC結果、fit、失敗logを
保存する。実行中も`matrix.json`と`report.json`を更新し、失敗と未完了を分母から
消さない。`report.json`は基準内の全pair、変形条件と各基準の全pair、距離6本の
全15pairを集計する。一定形状を返す対照は、非ゼロ変形の符号判定に失敗する。

基準は同じbodyで開始yawを少し変えた動画。独立した人間の撮り直しと同等ではない。
形状generatorと推定器が同じMHR familyのため、基本条件は有利な条件である。
局所変形・表面の波・左右非対称はMHR shape parameterを直接変更せず、geometryへ
加える。低frame rate、簡易lighting、単一body、rest pose、写真らしいtextureが
ない点も残る。cloth・呼吸・非対称は入力の外形自体を変えるので、単なる画像劣化に
よるfalse changeとは分けて解釈する。

回転不足は現行pipelineではmesh生成後のwarningであり、hard rejectionではない。
悪条件でmeshが出たことだけを成功と数えず、QCの網羅性・寸法のずれ・正しい拒否も
確認する。合成テストに合わせて採用基準や推定値を変更しない。

初期化時にprotocolとsource/config snapshotを非公開保存する。途中で実装が変わった
実験へ未処理ケースを継ぎ足すことは拒否する。orchestratorの同時実行をlockで防ぎ、
親が終了した後にworkerだけが完了した場合も、再開時にimmutable meshを再処理しない。

2026-10-02の33ケースは29件復元、4件QC拒否、未完了0。基準3本の最大差は
5.327 model mm。既知変化15pairの方向一致は14、最大nullの3倍超かつ方向一致は0。
一定meshの対照は0/15通過。距離の全15pairの最大対称差は1.986%。
遮蔽の2ケースは拒否されず、基準平均との対称差が11.04% / 60.73%になった。
複数人物・頭の切れ・無人は拒否、回転不足は網羅性flag不合格。複合劣化は人物未検出
で拒否された。これは単一合成bodyでの探索的な結果で、人体への採用根拠ではない。

29件の保存推定を調べると、初期shapeはusable frameの中央値と一致した。
+4cm相当の変形はbaseline_17比較で初期7.17 / fit後12.05 model mmの反応だった。
初期推定から過小反応があり、fitは反応を増やすが不足する。
通常条件のfit残差は3.62px、遮蔽は13.64 / 33.05pxだった。現在の
「fit前より改善した」判定だけでは不適切な復元を拒否できない。
残差の拒否閾値を採用するには独立した条件・body・実撮影での検証が必要。

## 修正候補の比較

同一動画を新しいstoreへコピーして比較する。以前のmesh/profileはコピーしない。
`synthetic_replay`へ渡したoptionsはmanifestにも保存し、実行中の変更を拒否する。
同じ入力の再処理なので、独立した撮り直しの件数には加えない。

```bash
uv run python -m body_scan.synthetic_replay \
  --source "$HOME/body-scan-private/synthetic-lab-new" \
  --out "$HOME/body-scan-private/synthetic-ablation-camera" \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json" \
  --options "$HOME/body-scan-private/camera-options.json" \
  --cases baseline_17 baseline_23 baseline_41 change_+0.040
uv run python -m body_scan.synthetic_lab run \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json" \
  --out "$HOME/body-scan-private/synthetic-ablation-camera"
```

`camera-options.json`の例。これはrendererの640×640、焦点距離430px専用であり、
実カメラへ流用しない。現在の輪郭投影は中央の主点・正方画素のみ対応する。

```json
{
  "camera": {
    "width": 640, "height": 640,
    "fx": 430, "fy": 430, "cx": 320, "cy": 320,
    "provenance": "declared_synthetic_renderer_intrinsics"
  }
}
```

腹部重視の比較は`"region": "abdomen", "region_weight": 8, "shape_weight": 0.1`
を追加する。推定されたcanonical anatomyのhip〜shoulderの10〜65%の帯域を使い、
全身のChamfer lossを維持しながら該当する点の重みを増やす。領域indicesは
最初の初期推定から固定し、正解meshの領域を参照しない。

QCの比較は次を追加する。

```json
{
  "qc": {
    "mask_support": true,
    "require_rotation": true,
    "max_relative_fit_error": 0.02
  }
}
```

mask supportは推定torso中央の3箇所に人物maskがあるかを確認する。
衣服・推定された隠れた形状が正しいことを証明する判定ではない。
回転の8bin中6bin以上かつ最大gap90度以下を要求し、最終残差を人物maskの
高さで割って判定する。重み付けfitでもQCには全身の重みなし残差を使う。
2pxのrobust-loss下限を含むので、この値は人体の寸法誤差ではない。
残差で拒否してもfit diagnosticsは保存し、mesh/profileは発行しない。
閾値・領域・重みは探索用であり、通常の撮影UIには有効化しない。

`research_options`のhashを含むmethod IDを使い、異なる候補のprofileを共有しない。
比較は方向一致、変化なしのばらつきの3倍を超えるか、距離、悪条件でのfalse changeと
拒否を確認する。計画されたpair数も残し、拒否したケースを除いて成功率を上げない。

```bash
uv run --with matplotlib python -m body_scan.synthetic_comparison \
  --variant "original=$HOME/body-scan-private/synthetic-lab-new" \
  --variant "camera=$HOME/body-scan-private/synthetic-ablation-camera" \
  --out "$HOME/body-scan-private/synthetic-comparison"
```

出力先はGit外に先に作成する。比較処理は入力hashの一致と未完了0を要求する。
元の実験のanalysisを保存し、比較JSON・Markdown・PNGを別に書き出す。

## 2026-10-03の比較結果

カメラ内部パラメータ、腹部の重み8、shape正則化0.1→0.01、16→48フレーム、
mask support・回転・残差QCを順に比較した。比較対象113回は96件復元、17件QC拒否、
予期しない失敗0、未完了0。異なる動画hashは44で、同一入力の再処理を含む。
以前の例外停止したQCと途中終了した実験も保存し、修正版とは混ぜていない。

最も反応が良かった48フレーム候補と元の実装の比較。単位は合成モデル上のmm。

|項目|元の16フレーム|48フレーム候補|
|---|---:|---:|
|体型1の基準3本の最大差|5.327|2.788|
|体型1の既知変化の方向一致|14/15|15/15|
|体型1の方向一致と最大差の3倍超|0/15|10/15|
|体型2の基準3本の最大差|11.073|1.582|
|体型2の方向一致と最大差の3倍超|0/9|9/9|

体型1では−20/＋20/＋40の全pairが余裕を確保したが、＋10は0/3、−10は1/3だった。
体型2の変形は−20/＋20/＋40のみ。48枚候補でも距離1.5/2.5の対称差3.488%、
ぼけのfalse change3.588%が残り、いずれも1%の実験基準に届かない。
大小の遮蔽と回転不足は両体型で拒否された。撮影条件19種の修正版QCは12件受理・
8件拒否。頭が画面外の座標処理で見つかった例外も修正し、正常なQC拒否になった。

カメラ修正だけでは体型1の基準差13.061mm、距離差5.243%へ悪化した。
設定の誤りを直せば測定精度も上がる、という仮説は支持されない。
腹部の重みと弱い正則化は反応を増やし、フレーム数の増加は開始角度による差を減らした。
体型1の＋10変形の投影変位は最大0.378px、肩/腰の推定誤差は平均約2pxだった。
これは誤差と変形の大きさの比較であり、単独で原因を断定する証拠ではない。

48フレームは各動画に元からある48枚を全て使い、新しい動画を生成して水増ししない。
異なるsamplingのengine IDとstoreを使い、同じinference lock・メモリ制限を維持した。
体型1の48枚処理の中央値は113.8秒。CPU上のsegmentation・MHR・fitを含む。
最初の2本の基準差だけで最終的なばらつきを評価せず、3本の全pairを使用する。

体型2は最初の候補には事前に用意した別体型だが、48枚の追加比較では以前の結果を
見た後に同じデータを再利用している。探索的な確認であり、未使用データでの最終検証
とは扱わない。2体型ともMHR由来、固定pose、簡易rendererで、人体のcm精度は未検証。
通常のSoma測定への採用には不足する。人間の撮り直し・衣服・姿勢・距離の検証を残す。

元の16枚の既定処理は、同一入力で旧版の頂点座標と完全一致した。
今後のsource snapshotにはsamplerの`vision.py`も含める。古い実験に実行時configの
snapshotがない場合は、保存したframe outcomesから枚数を確認し、その制約を明記する。

## 層ごとの改善・追加検証

[改善TODOと採用判断](improvement-plan.ja.md)に上流からの比較結果を記録した。
追加の探索は保存SAM推定の81回、通常pipelineは28回（20件復元、8件QC拒否）。
fitと出力で骨格サイズが異なる不整合を修正した研究候補で、同じ体型2の距離差は
3.488%から0.352%になった。別体型3では1.034456%のため1%基準は未達。
＋10 model mmの変化は3.45〜5.23しか推定されず、ばらつきの3倍を超えなかった。

人物だけのぼけ・noiseによる従来の品質判定の抜けを修正した。人物境界のnoise補正済み
指標で、閾値選択に使わない画像の良好1件を受理、悪条件4件を拒否した。
48枚を維持し、16/32枚のfit、yaw追加、輪郭再抽出は候補へ採用しなかった。

CPU pool上書きの修正後の既定v1は過去の出力に対して周長差0.2685 model mmがある。
上の完全一致は修正前の確認であり、現在も完全一致するという意味ではない。
現在の研究候補を同入力で再実行すると頂点は完全一致した。実人体の精度は未検証。

保存済み合成入力を研究設定なしで再実行する場合は`--options`を省略する。
空の研究設定`{}`を渡す実験とはmethodを区別する。

```bash
uv run python -m body_scan.synthetic_replay --help
uv run python -m body_scan.synthetic_refit --help
```

実験ごとの入力・モデル・実装hashとQC診断をGit外へ保存し、例外をQC拒否扱いしない。
