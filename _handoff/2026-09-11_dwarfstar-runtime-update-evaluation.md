# DwarfStar runtime更新評価 — 現行版を継続

2026-09-11、Qwenモデルを変更せずDwarfStarだけを更新する依頼に対し、別checkoutでビルドと比較を実施した。候補は品質チェックを通ったが、M4 Maxで生成速度の低下が再現したため本番には採用しなかった。現行バイナリ・モデル・設定を維持し、停止したChronovisorサービスは復旧済み。**runtime更新そのものは未適用**。

## 比較対象と固定条件

- 現行: `ivanfioravanti/ds4-metal` `ffd85d426313ace6dae805e9f2fb4424d5b427fd`。
- 候補: 同リポジトリの `qwen3.8-flash-next` ブランチ `6c1e83672650e1bae9fe460a05c7f131d93f055e`。
- Apple M4 Max、128 GiB。全モデルを同時に2プロセスでロードせず、順次比較。
- モデル: `Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf`（50,343,093,376 bytes）、外部PLE: `Qwen3.8-Flash-Next-PLE-Q4_1.gguf`（32,000,157,440 bytes）。新しい量子化モデルは取得していない。
- 本番LaunchAgentと同じ `--metal --ctx 262144 --prefill-chunk 1024 --power 100 --mtp --mtp-exact-sampling`。生成測定は temperature=0、seed=42、thinking無効。
- 短文品質32件、4K/16K needle、384-token生成3回、20,792-token実資料生成、Chronovisorの既存native adapter/DecisionRouterを使う4ケースを再利用。実資料は既存のローカル検証用snapshotを使用し、本番wikiへ書き込んでいない。

## 検証結果

候補のMetal build、CPU build、server unit test、Qwen Metal kernel testは成功。CPU build後にMetal版を強制再ビルドし、最終Metalバイナリでserver/kernelテストを再実行した。CPUで巨大モデルの推論は行っていない。

短文品質は両版とも27/32件成功し、**32件すべての回答文字列が一致**。既知の不一致は `percent`, `remainder`, `set`, `precedence`, `jp_order`。needleは各2/2成功。実資料は両版とも形式検証に合格し、20,792 input / 2,045 output tokensで回答も完全一致。native acceptanceは両版とも4/4件でschema・意味効果が一致した。網羅的な品質保証ではない。

| 測定 | 現行 | 候補 |
|---|---:|---:|
| 4K needle TTFT | 6.73秒 | 7.01秒 |
| 16K needle TTFT | 26.46秒 | 27.07秒 |
| 生成速度中央値・最初の3回 | 55.32 token/s | 50.06 token/s |
| 生成速度中央値・逆順で再測定3回 | 53.98 token/s | 51.20 token/s |
| 実資料 TTFT | 40.19秒 | 47.48秒 |
| 実資料 全体時間 | 76.44秒 | 86.69秒 |

最初の順序は現行→候補。速度差の再現確認だけ候補→現行の順序で追加した。生成速度の低下は9.52%、逆順でも5.16%。実資料の処理時間は約13.4%増加した。初回両armでSwapoutsの増加は0。今回は同じ設定で性能が維持されるという採用条件を満たさなかった。

ソースレビューでは旧IQ2/MXFP4Downの構造的互換性に明確な問題は見つからなかった。新たに既定となったMTP GPU argmax、state swap restore、PLE prefetch方針の変更があり、速度差への寄与は未特定。`DS4_QWEN4_MTP_GPU_ARGMAX=0`等による挙動変更の切り分けは実施していない。既定設定での更新評価の範囲を越えて、独自のruntime調整や上流修正は加えていない。

## 復旧と証跡

Ingestのcross-process leaseを取得してbatch完了の境界を確保し、model resource leaseで推論終了を待ってからサービスを停止した。2回の比較のfinally処理で、元々enabledだった11サービスとDwarfStarを復旧。config.tomlとDwarfStar plistのSHA256、引数、モデル・PLEのsize/mtime/inodeが比較前後で不変であることを確認した。82GBの重み全体を再hashする検証は省略した。

復旧後のDwarfStarはPID `70206`、実行ファイルは元の `/Users/trafficsign/.local/share/dwarfstar/runtime/ds4-server`。元のcommitとbinary SHA256が一致し、`/v1/models`および生成canary `READY`が成功した。Dashboardの実ポート8765はHTTP応答があり、current authorityはready。集約healthはstaleだったため、それだけを復旧判定に使っていない。semantic/rerankerは数秒以内に更新されたstatusでready=true、プロセス生存を確認した。rerankerの履歴last_errorには `reranker_unavailable` が残るため、全機能の無故障を主張するものではない。

- 現行binary SHA256: `579696f76eb8ee6acfcff8e3810926c8b143674d6485e8526a6dadc4aa70e2c5`
- 候補binary SHA256: `aea3a83dabaf6569f1d38c18fbc8669835b0b7ffa03bd1c36f61cad7e37caf43`
- 非公開の詳細証跡・比較script: `/Users/trafficsign/projects/sandbox/dwarfstar-update-20260911/`。`results/decision-summary.json`、`comparison.log`、`recheck.log`、`build-logs/`、`results/final-restoration.json`を保持。実資料と生成本文はこのローカル領域に留め、Gitには含めない。
- 未使用となった候補checkout/build（約189MB）と未実行の切替scriptは、候補プロセスが停止済みでモデルファイルを含まないことを確認して削除した。既存runtime・既存モデル・既存の検証fixtureは保持した。
- 無関係な作業ツリー変更は保持。この報告だけをcommitし、pushは行わない。
