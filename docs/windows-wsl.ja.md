# Windows / WSL2 + NVIDIA ランタイム

WSL2のLinuxファイルシステムにチェックアウトし、Linux側のuv/Pythonを使う。
Windows側ではNVIDIAドライバを用意し、WSLで`/usr/lib/wsl/lib/nvidia-smi`
がGPUを認識することを確認する。Windowsブラウザのloopback URLで撮影画面を
開けば、ブラウザがWindowsのカメラを使う。UIはHostを厳密に検査するので
`serve`が表示する`http://127.0.0.1:3212`を使う。

```bash
uv sync --locked --extra vision --python 3.11
# モデルの承認前にも依存・通信隔離・実GPUカーネルを確認できる
uv run body-scan setup-linux --device cuda --dependencies-only \
  --out "$HOME/body-scan-private/wsl-cuda-dependencies"
BODY_SCAN_TEST_LINUX_SANDBOX=1 uv run python -m unittest discover -v
```

Hugging Faceの[モデル](https://huggingface.co/facebook/sam-3d-body-dinov3)への
利用承認後、WSL内でログインし、別の新規ディレクトリへモデルを配置する。
トークンは対話入力に使い、Git・チャット・コマンド引数に入れない。

```bash
uvx --from huggingface_hub hf auth login
uv run body-scan setup-linux --device cuda --frames 16 \
  --out "$HOME/body-scan-private/wsl-cuda-runtime"
uv run body-scan doctor \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json"
uv run body-scan smoke-runtime \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json" \
  --out "$HOME/body-scan-private/smoke-new-01"
uv run body-scan serve --port 3212 \
  --data-root "$HOME/body-scan-private/wsl-scans" \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json"
```

`--device`は必須で、`cuda`または`cpu`を明示する。CUDAは固定Torch 2.10.0 /
torchvision 0.25.0のCUDA 12.8 wheelを使う。DINOエンコーダーのみCUDA/float16
へ配置し、SAM/MHRのhead、人物segmentation、輪郭fitはCPUのまま実行する。
CUDAが使えない、GPUカーネルが失敗する場合はCPUへ逃げず停止する。

推論はLinuxのuser/network namespaceで通信隔離する。host localhostの
実際に待ち受けているsocketと外部接続を起動前に検査する。`unshare`欠落、
権限不足、検査失敗ならモデル処理を開始しない。環境変数によるoffline設定も
継続する。推論supervisor停止時にはchildも終了する。

既存のhost RAM監視、単一実行lock、モデルchecksum、非公開保存を維持する。
WSLではRSS上限8192 MiB / 空きRAM最低2048 MiBを既定にし、
`--max-rss-mb` / `--min-available-mb`で明示変更できる。Macの6144/1200 MiB
は保持する。実機では旧6144 MiB上限がモデル読込の6167 MiBで発動した。
CUDA allocatorは既定でGPU容量の50%に制限する。完了時の
`reconstruction.json`にbackend、経過秒、peak RSSとCUDA allocated/reserved
を記録する。GPU driverや他プロセスのメモリはCUDA allocatorのpeakに含まれない。

両setupコマンドは新規ディレクトリ専用。再試行で既存データを上書きしない。
推論結果の比較には同じprivate動画、16フレーム、同じmodel/source revisionと
fit設定を使い、Mac/WSLで別のstoreを用意する。backendを変えたprofileを
共有しない。CUDAカーネル確認はSAM全推論や身体変化の精度の証拠ではない。
Macで同じ動画を処理した比較と実身体の採用ゲートは別途必要。

Soma [#55](https://github.com/Andyyyy64/soma/issues/55) のWindows実行対応は
開発中。Somaへの製品統合は[#54](https://github.com/Andyyyy64/soma/issues/54)
の別工程であり、このチェックアウトの変更だけでは採用完了にならない。

2026-10-02のWSL検証: 外部runtimeで57テストが成功。通信隔離中の合成入力SAM/MHR
smokeは11.98秒、peak RSS 6286 MiB、CUDA allocated 1689 MiB / reserved 1754 MiB。
これは1フレームの実モデル接続確認で、16フレームの実動画処理やMac比較ではない。
working-treeのpublic-file auditは成功。upstream mainに既存の非匿名commit identityが
あるため`audit --history`は未解決の失敗であり、この作業では履歴を書き換えない。
