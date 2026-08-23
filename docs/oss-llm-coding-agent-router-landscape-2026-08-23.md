# OSS LLM／coding-agent router landscape（2026-08-23）

調査日: 2026-08-23（JST）
調査方法: 公式 GitHub リポジトリ、公式 README・ドキュメント、LICENSE、公式 release／issue だけを確認した。二次記事の比較表やベンチマークは根拠に使っていない。製品自身の README にある性能・精度の数字は、独立検証ではなく「公式の主張」として扱う。

## 結論先出し

一般的な vendor-neutral OSS 一つで「Codex のフル機能サブエージェント（filesystem、tools、child delegation）を保ちつつ、Responses/Chat/Anthropic 変換、provider registry、fallback/health、意味的モデル選択」までを同時に満たすものは少ない。だが今回のスコープには例外的に、Codex 専用の最有力候補 **`duolahypercho/codex-router`** がある。MIT の self-hosted router で、Codex native model catalog、provider/model registry、OpenCode Go の Chat/Messages/Responses catalog、credential isolation、control/doctor、quota-aware failover、`apply_patch` bridge、namespace tool relay、native encrypted child-task relay の境界を既に持つ。したがって第一 PoC は CCR ではなく `codex-router` である。

ただし、`codex-router` 自体も「Codex child を生成する別 harness」ではない。Codex の native spawn を保ったまま payload/tool を relay する Codex 専用 integration であり、semantic complexity router ではない。また current README の request flow は Codex Responses → 内部 LiteLLM → provider なので、現在の自作構成がこの系統の fork/拡張である場合、LiteLLM を消す単純な置換というより upstream hardening／差分削減として評価する必要がある。[repo](https://github.com/duolahypercho/codex-router)・[architecture/request flow](https://github.com/duolahypercho/codex-router#how-routing-works)・[v2 agent gate](https://github.com/duolahypercho/codex-router/blob/main/v2_agent/README.md)

優先順位は次のとおり。

1. **第一 PoC は `duolahypercho/codex-router`**。MIT、macOS の Homebrew／per-user service／tray、Codex 専用 Responses 入口、モデル catalog、provider selection、caller capability、doctor、quota-aware failover/control center、native tool namespace relay、`apply_patch` custom bridge を一体で扱う。v2 の exact route には streamed Responses、forced function call、暗号化 child payload relay、同一 child への follow-up の 5-check gate がある。[README](https://github.com/duolahypercho/codex-router)・[v2 gate](https://github.com/duolahypercho/codex-router/blob/main/v2_agent/README.md)・[release](https://github.com/duolahypercho/codex-router/releases)

   重要な但し書き: v2 は route ごとの証明であり、2026-08-23 に親セッションが確認した config では Kimi/Grok の 6 exact route だけが grandfathered。DeepSeek／OpenCode Go は catalog に載っていても native child の v2 認証済みとはみなさない。DeepSeek exact route／OpenCode Go exact route は、PoC と `v2_agent` application／upstream registry change が必要である。

2. **次点の gateway PoC は `musistudio/claude-code-router`（CCR）**。MIT、macOS のローカルアプリ／Node gateway、Codex の Responses と ordinary `apply_patch` bridge、OpenAI Chat/Responses・Anthropic Messages、ordered fallback を一つのローカル入口にまとめている。現在の LiteLLM＋`sitecustomize.py` のうち、protocol adapter／provider registry／fallback を置き換えられる可能性が高い。ただし current CCR main の `codex-multi-agent-bridge` は `multi_agent_v1` 固定で、native encrypted child-task relay は確認できない。Codex child spawning は CCR が提供するのではなく、Codex が引き続き担当する。[リポジトリ](https://github.com/musistudio/claude-code-router)・[provider protocol](https://github.com/musistudio/claude-code-router/blob/main/docs/src/content/docs/en/guides/provider.md)・[Codex routing](https://github.com/musistudio/claude-code-router/blob/main/docs/src/content/docs/en/configuration/routing.md)・[release](https://github.com/musistudio/claude-code-router/releases)

3. **gateway を Go の単体プロセスに寄せるなら `maximhq/bifrost`**。Apache-2.0 の core、公式の Codex CLI 設定（`/openai/v1`＋`wire_api = "responses"`）、OpenCode 設定、provider/model catalog、CEL の conditional routing、retry/fallback がある。CCR より「汎用 AI gateway」寄りで、semantic routing と Codex agent spawning/native encrypted relay は持たない。[リポジトリ](https://github.com/maximhq/bifrost)・[Codex](https://docs.getbifrost.ai/cli-agents/codex-cli)・[OpenCode](https://docs.getbifrost.ai/cli-agents/opencode)・[routing rules](https://docs.getbifrost.ai/providers/routing-rules)・[fallback](https://docs.getbifrost.ai/features/retries-and-fallbacks)

4. **Codex 側の agent role／model policy を整理するなら `oh-my-openagent`（旧 oh-my-opencode）／LazyCodex**。Codex Light Edition は Codex plugin system、`~/.codex/agents/`、Codex 自身の spawn/collaboration surface を使うため、現在の per-model agent TOML に近い。しかし core は OSI OSS ではなく Sustainable Use License（SUL-1.0）の source-available、LazyCodex の MIT は薄いインストーラに過ぎない。protocol gateway の代替ではない。[本体](https://github.com/code-yeongyu/oh-my-openagent)・[license](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/LICENSE.md)・[agent model matching](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/docs/guide/agent-model-matching.md)・[LazyCodex](https://github.com/code-yeongyu/lazycodex)

5. **意味的 routing は後付けの別レイヤー**。Apache-2.0 の [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) は Responses ingress、Chat ingress、Anthropic backend conversion、YAML policy を備えるが、Responses は内部で Chat Completions に変換され、cross-model fallback は現状の実装ではなく proposal である。[Plano（旧 Katanemo Arch）](https://github.com/katanemo/plano) は alias／preference／cost／latency／429・5xx fallback と Responses state を持つが、既定では hosted の Plano-Orchestrator に依存し、agentic loop では model affinity を明示的に有効化しないと request ごとにモデルが変わり得る。[vLLM API](https://vllm-sr.ai/docs/v0.3/api/router/)・[fallback proposal](https://vllm-sr.ai/docs/proposals/model-execution-fallback/)・[Plano routing](https://docs.planoai.dev/guides/llm_router.html)・[Plano state](https://docs.planoai.dev/guides/state.html)

したがって現実的な第一案は次である。

```text
Codex（tools / filesystem / native child delegation は維持）
  -> codex-router（native namespace/encrypted relay + catalog + control）
       -> LiteLLM / provider adapter（現 upstream flow）
  -> OpenCode Go / その他 provider
```

`codex-router` の current README は内部に LiteLLM を置く構成を明示している。このため、現在の自作が同じ `:4202 -> :4200 -> provider` 系なら、まず upstream の installer／catalog／relay／control を採用して差分を減らし、protocol adapter の LiteLLM を残す案が最小である。LiteLLM を直接外したい場合だけ、shadow endpoint で CCR/Bifrost を別 PoC として比較する。

今回の優先順位には、親セッションの実測も効いている。Desktop `0.149.0-alpha.4.1` では、外部モデル role を指定した v2 child の metadata が `model_provider=openai` のまま、`The 'ox-alpha-free' model is not supported when using Codex with a ChatGPT account.` で終了した。公開 `rust-v0.148.0` は role が provider を明示しない限り current provider を保持し、partial fork の v2 test も role の `model_provider = "ollama"` が child snapshot に入ることを検査している。したがって、これは外部 route 不可能の証明ではなく、Desktop alpha の適用差分／回帰として扱う。role TOML だけに依存せず、native catalog と encrypted relay を持つ `codex-router` を先に検証する理由である。[public role implementation](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/agent/role.rs)・[v2 provider override test](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/tools/handlers/multi_agents_tests.rs#L902-L956)

semantic router は最初から挟まず、AGENTS policy から明示的な logical model alias／header を渡す。semantic classifier を入れる場合も「ユーザーの新しい turn の開始時だけ分類し、tool loop 中は model を pin」する。CCR、Bifrost、LiteLLM を三重に連結して同じ Responses↔Chat 変換を繰り返すのは避ける。

## 現在の自作スタックを責務で分解

| 責務 | 現在の実体 | 失うと困る性質 | OSS 候補が代替し得る範囲 |
|---|---|---|---|
| Agent harness | Codex CLI／Codex custom agent TOML | tools、filesystem、MCP、full-capability、child delegation | gateway は代替しない。OMO Light／Ruflo／OpenCode／OpenHands は harness 候補だが、Codex の実行意味論を変える |
| Codex ingress | custom provider `opencode_go_gateway`、`http://127.0.0.1:4141/v1`、`wire_api = "responses"` | Codex が期待する Responses の request/stream/event | CCR、Bifrost、LiteLLM、vLLM SR、Plano が入口を提供し得る |
| Protocol adapter | LiteLLM 1.96.2、Responses→OpenCode Go の Chat Completions／Anthropic-compatible | tool call transcript、streaming、reasoning、custom tool | CCR は Codex `apply_patch` を専用 bridge。Bifrost は provider adapter。vLLM SR／Plano は別の変換層を持つ |
| Transcript repair | `sitecustomize.py` の未回答 tool call 除去、tool adjacency 補修 | upstream が拒否する履歴を防ぐ | 代替製品で同じ replay test を通すまで削除不可 |
| Model/provider registry | LiteLLM の deployment/provider 設定と `/models` 補修 | alias、model list、credential、upstream path | CCR の provider/profile、Bifrost の model catalog、LiteLLM が該当 |
| Reliability | LiteLLM の retry/fallback/cooldown と AGENTS の Luna→DeepSeek→Terra→Sol 選択 | 429/5xx/timeout 時の制御、stream 開始後の二重実行回避 | CCR／Bifrost は retry/fallback。vLLM SR は現状同一 backend cluster の retry が中心 |
| Task／role policy | 親 Codex の AGENTS policy、複数 per-model agent TOML | role ごとの model、delegation の境界 | OMO Light、Ruflo、OpenCode agent config は policy を簡素化し得るが、Codex の native child semantics との比較が必要 |
| Semantic routing | 現在はなし | complexity／task type による自動選択 | vLLM SR、Plano、RouteLLM、Ruflo（公式には intelligent routing を主張）。ただし別レイヤーであり、まず不要 |

この分解から、router を入れればサブエージェントが自動的に呼べるようになる、とは言えない。router は通常 HTTP request の model／provider を変えるだけで、Codex の `spawn_agent`、child session、tool permission、filesystem sandbox を生成しない。

### 現行 `codex-router`／v2 relay の確認事項（親セッションでの実測）

OSS 比較だけでなく、現行の自作 router が守ろうとしている境界も採用条件に入れた。親セッションが 2026-08-23 に確認した current main の事実は次のとおりである。

- `codex-router` main は `e8c428d`。[OpenCode catalog research](https://github.com/duolahypercho/codex-router/blob/e8c428d/docs/research/opencode-catalog-2026-08-21.md) は「Every OpenCode model stays conservative v1 unless marker-return/encrypted relay/same-thread proof」とし、OpenCode model を安全側に扱う方針を取る。
- `v2_agent` README は provider/model route ごとに 5 checks を要求するが、config 上の v2 route は現状 Kimi/Grok の 6 route のみ。DeepSeek／OpenCode Go は認証済み route ではない。
- 現行 `check_gateway.py` は ordinary function tool と plaintext marker の保持を確認するだけで、native encrypted child-task relay、namespace tool round-trip、same-thread encrypted payload を検査していない。
- CCR main `1347c868`（2026-08-22）は `codex-multi-agent-bridge` が `multi_agent_v1` 固定で `apply_patch` bridge を持つ。CCR の `encrypted_content` は session affinity／context archive 用で、現行 `codex-router` と同等の native encrypted child payload relay は確認できない。
- 公開 Codex Rust `rust-v0.148.0` の role layer は、role TOML が `model_provider` を明示しない限り current provider を保持する。さらに v2 partial-fork test は role の model、`model_provider = "ollama"`、reasoning effort が child snapshot に適用されることを直接検査している。[role implementation](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/agent/role.rs)・[role tests](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/agent/role_tests.rs)・[v2 provider override test](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/tools/handlers/multi_agents_tests.rs#L902-L956)
- これとは別に、親セッションの sanitized rollout observation（2026-08-23 11:30）では、`opencode_go_ox_alpha` を role に指定した v2 child を 2 回 spawn した際、metadata は `agent_role=opencode_go_ox_alpha`、`model_provider=openai`、`multi_agent_version=v2` となり、各 child が `The 'ox-alpha-free' model is not supported when using Codex with a ChatGPT account.` で即時終了した。child 引数は暗号化 payload であり、値は転載しない。これは Desktop `0.149.0-alpha.4.1` の role/provider 適用が公開 Rust source と異なる、または alpha 固有の回帰である可能性を示す親セッション内の観測であり、upstream の原因確定ではない。
- この観測は「provider TOML を書けば外部 child route になる」と仮定できないことを示す。外部 route では role/provider の解決に加え native encrypted child payload relay が必要になるため、`codex-router` の openai base URL＋merged native catalog＋relay は、この Desktop 側の role/provider 問題を迂回できる候補として評価する。ただし exact route の v2 proof は別途必要である。

したがって CCR の「Codex 対応」は、今回の採用条件のうち **Responses／Chat／Anthropic protocol と `apply_patch` の適合**を示すが、**native encrypted child-task relay の代替証拠ではない**。CCR／Bifrost の PoC は v1 の ordinary function path だけで合格にせず、現行 router が conservative に要求する marker-return、encrypted relay、same-thread proof を別の acceptance fixture として含める。DeepSeek／OpenCode Go の route は、公式製品 docs の capability 表だけでは認証済みとみなさない。

## 候補一覧

### Coding-agent／orchestration

| 候補（identity） | license／self-host | routing／agent surface | API・fallback | Codex への適合と置換範囲 | 判定 |
|---|---|---|---|---|---|
| [duolahypercho/codex-router](https://github.com/duolahypercho/codex-router) | MIT。Codex 専用の macOS/Linux local service、Homebrew、tray/control、per-user state。current main `e8c428d`、latest release `v0.4.0-beta.4`（2026-08-15）。 | provider/model registry、explicit picker/curation、credential-aware publication、Codex namespace/tool inventory、native encrypted child-task relay。semantic complexity classifier ではない。 | Responses ingress。OpenCode Go を `opencode-go`（Chat）、`opencode-go-messages`（Anthropic Messages）、`opencode-go-responses`（Responses）に分けて catalog。quota-aware 402/long-429 failover、health/doctor、rollback。 | 現在の Codex harness と native child delegation を維持し、relay／catalog／control を置換候補にする。README の flow は内部 LiteLLM を使うため、protocol adapter まで消えるとは限らない。DeepSeek/OpenCode Go exact child は v2 application が必要。 | **最有力／第一 PoC** |
| [Claude Code Router](https://github.com/musistudio/claude-code-router) | MIT。Node gateway と macOS/Windows/Linux desktop。既定 `127.0.0.1:3456`。 | static route、body/header 条件、dynamic script、model-chain fallback。Claude subagent には prompt tag routing があるが、Codex child の semantic routing とは別。 | OpenAI Chat/Responses、Anthropic Messages、Gemini。Codex の freeform `apply_patch` を `virtual_apply_patch` にして upstream function に変換し、応答を Codex custom tool に戻す。retry/failover。 | Codex の harness／spawn は保持。LiteLLM＋`sitecustomize` の protocol/provider/fallback の大部分を置換候補。OpenCode Go の実接続と reasoning/tool replay は未検証。native encrypted child relay は current main で確認できない。 | **次点 gateway PoC** |
| [oh-my-openagent](https://github.com/code-yeongyu/oh-my-openagent)／[LazyCodex](https://github.com/code-yeongyu/lazycodex) | 本体は SUL-1.0（source-available、厳密な OSS ではない）。LazyCodex installer は MIT。ローカル Codex plugin。 | agent/category ごとの model fallback と runtime fallback、role chain。main prompt の自動 complexity routing は少なくとも公式 feature request では未確定。 | gateway／protocol adapter ではない。Codex Light は Codex native tools/spawn を使用。 | per-model TOML／AGENTS の整理に近い。LiteLLM、CCR、Bifrost の代わりにはならない。 | **policy 候補。ただし license 注意** |
| [Ruflo（旧 Claude Flow）](https://github.com/ruvnet/ruflo) | MIT。Node CLI/MCP。macOS/Linux/WSL/Git Bash。 | 公式 README は hooks、swarm、100+ agents、multi-provider smart routing、failover、self-learning を掲げる。CLI の `hooks route` と agent spawn がある。 | 314 MCP tools／provider routing を主張するが、Codex Responses↔Chat/Anthropic の wire-level adapter の公式保証は見当たらない。 | Claude Code plugin が中心。Codex integration は README に明記されるが、Codex native child delegation の置換・共存を実測要。大規模な別 harness を導入するため変更範囲が大きい。 | **adjacent。広すぎるので第一選択ではない** |
| [CodeRouter](https://github.com/Code-Router/CodeRouter) | MIT。Node 24+ の CLI／MCP と Electron desktop。2026-08-23 時点で 2 stars／1 fork の新興 project。 | prompt の cognitive shape、task type、過去 run で route を選び、plan／agent／debug／review workflow と fixer→reviewer handoff を持つ。 | OpenAI、Anthropic、OpenRouter、DeepSeek、Groq、Ollama adapter。Codex／Claude Code からは MCP tool として呼ぶ。 | Codex の native provider／child relay を置換する透明 gateway ではなく、別 worktree で実行する meta-agent。設計比較には有用だが、成熟度と変更面積から第一 PoC にはしない。 | **新興の別 harness** |
| [OpenCode](https://github.com/anomalyco/opencode) | MIT。CLI と desktop、macOS/Homebrew。 | build（full access）、plan（read-only）、`@general` 等の primary/subagent。agent TOML/JSON と task permission。 | provider は `@ai-sdk/openai-compatible`（Chat）または `@ai-sdk/openai`（Responses）。gateway の provider pool／fallback は主責務ではない。 | OpenCode 自体を Codex の代わりにするなら harness 変更。OpenCode を gateway にする製品ではない。 | **代替 harness、router ではない** |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | Apache-2.0。ただし公式 repo は 2026-05-15 に archive、extension shutdown。 | Code/Ask/Architect/Debug/Orchestrator mode、mode ごとの model。Orchestrator は `new_task` で委譲。 | provider は model-agnostic だが gateway ではない。archive 後の新規運用リスク。 | Codex harness を保持しない。 | **除外** |
| [Continue](https://github.com/continuedev/continue) | Apache-2.0。公式 README が repository は no longer actively maintained、final 2.0.0 と明記。 | chat/edit/apply/autocomplete/embed/rerank 等の model role。 | provider／role 設定はあるが、gateway の fallback／Codex child spawn ではない。 | active router として採用しない。 | **除外** |
| [Aider](https://github.com/Aider-AI/aider) | Apache-2.0。ターミナル agent。 | architect model＋editor model、`weak_model_name`、LiteLLM の追加パラメータ。 | agent 内の model 分担はあるが、汎用 proxy、Codex child delegation、Responses bridge ではない。 | Codex の置換 harness としては別製品。 | **比較用** |
| [OpenHands Agent Canvas](https://github.com/OpenHands/OpenHands) | MIT。self-hosted local/REST control center。Codex、Claude Code、Gemini、ACP agents を backend として扱う。 | 複数 agent backend、automation、Agent Server。 | `any LLM`／agent backend の制御面だが、LiteLLM 相当の wire gateway ではない。Node 22.12＋uv、sandbox/Docker を含み得る。 | 親 orchestration を置換する場合の候補。Codex の native subagent semantics を維持する最小変更案ではない。 | **大規模な別案** |

補足: `oh-my-opencode` は canonical repository が現在 `oh-my-openagent` に rename 中で、npm／README には旧称が残る。`lazycodex-ai` は本体機能を実装する gateway ではなく、OMO plugin を Codex に入れる distribution layer である。license を「LazyCodex が MIT だから本体も MIT」と扱ってはいけない。

### Gateway／proxy

| 候補（identity） | license／self-host | routing／reliability | API・tool | Codex／macOS／置換範囲 | 判定 |
|---|---|---|---|---|---|
| [LiteLLM](https://github.com/BerriAI/litellm) | 非 enterprise core は MIT。self-host proxy。100+ model provider。現在利用中。 | weighted、rate-limit、latency、least-busy、lowest-cost、custom strategy、fallback、cooldown、context-window handling。 | `/chat/completions`、`/responses`、`/messages`、streaming、tools/MCP を公式 docs に記載。 | 既存構成の最小変更。新製品ではないが、安定性と機能の基準線。 | **残すのが最小リスク** |
| [Bifrost](https://github.com/maximhq/bifrost) | core Apache-2.0。Go gateway、Node launcher、Docker、UI。 | provider/model catalog、CEL 条件、weighted/budget/rate-limit、retry、key rotation、sequential fallback。adaptive LB 等は enterprise。 | OpenAI-compatible。公式 Codex は `/openai/v1`＋Responses、OpenCode は `/openai`／Anthropic endpoint。streaming、tools、multimodal。 | LiteLLM＋patch の gateway replacement の有力候補。Codex harness は維持。Go 単体運用は macOS に向く可能性があるが、実 RSS/起動時間は未測定。 | **第二 PoC** |
| [Portkey Gateway](https://github.com/Portkey-AI/gateway) | MIT。local Node gateway。Gateway 2.0 は pre-release。 | retry/fallback、load balance、conditional routing。 | provider gateway として OpenAI/Anthropic 等を統合。Responses／Codex tool fidelity は別途確認。 | provider/fallback は置換できる可能性があるが、CCR/Bifrost より Codex 直結の公式資料が弱い。 | **比較候補** |
| [Helicone AI Gateway](https://github.com/Helicone/ai-gateway) | current `ai-gateway` の raw LICENSE は GPLv3。README や旧 Helicone repo の Apache 表記と混同しない。Docker self-host と hosted/cloud。 | fastest/cheapest/reliable、BYOK/managed key、429/401/400/408/500+ failover。 | OpenAI-compatible syntax、100+ models。Responses／Anthropic／Codex tool round-trip は未確認。 | cloud/provider catalog が主。license と secret/telemetry 境界を確認するまで採用しない。 | **除外寄り** |
| [TensorZero](https://github.com/tensorzero/tensorzero) | Apache-2.0。公式 repo は 2026-06-12 に owner が archive/read-only。 | routing、fallback、retry、load balancing、tools/multimodal を掲げる。 | OpenAI SDK／gateway。 | active OSS としての採用候補から除外。 | **除外** |
| [Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway) | Apache-2.0。Envoy/Kubernetes 向け。 | gateway routing、provider integrations、Kubernetes/Envoy の運用。 | OpenAI-compatible と Anthropic-compatible の対応範囲を公式 supported-endpoints に記載。 | macOS 単機の軽量 proxy には過大。Codex child spawn はない。 | **infra 向け** |
| [Higress](https://github.com/higress-group/higress) | Apache-2.0、CNCF Sandbox。Istio/Envoy/Wasm 系。 | AI gateway、multi-model LB/cache/rate limit、MCP、protocol unification。 | 複数 AI provider／MCP。Responses の Codex tool parity は実測要。 | Kubernetes/Envoy の制御面が重い。 | **infra 向け** |
| [Apache APISIX AI Gateway](https://apisix.apache.org/ai-gateway/) | Apache-2.0。APISIX/etcd/Nginx の gateway。 | AI proxy、model routing、LB、retry/fallback、token limit、observability。 | `ai-proxy` 等。 | gateway としては強いが、Mac の single local process、Codex spawn、OpenCode Go の直接置換には過大。 | **infra 向け** |
| [Kong AI Gateway](https://docs.konghq.com/gateway/latest/get-started/ai-gateway/) | Kong Gateway OSS の AI Proxy と enterprise の AI Proxy Advanced を区別する必要がある。 | OSS は basic AI proxy。advanced LB/routing/retry 等は enterprise docs。 | OpenAI Chat/Completion/streaming の plugin docs。 | 企業 gateway の選択肢であり、全機能 OSS の単機 Codex router ではない。 | **比較用** |

### Semantic／learned model router

| 候補（identity） | license／self-host | routing basis | protocol／fallback | Codex への適合 | 判定 |
|---|---|---|---|---|---|
| [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | Apache-2.0。Docker/Kubernetes、CPU router、macOS/WSL2 quickstart。Envoy extproc data plane。 | YAML policy、signals、model capability、semantic classifier 等で Mixture-of-Models。モデル weights を router が持つのではなく backend を選ぶ。 | `/v1/chat/completions` と `/v1/responses` ingress。Responses は内部 Chat へ変換。Anthropic backend conversion（tools/tool calls/streaming）あり。現行 fallback は同一 cluster retry/outlier が中心、cross-model fallback は proposal。 | semantic decision layer として有力だが、Codex child spawn なし。Responses の ID、reasoning、tool transcript の lossless replay が必須。Envoy/Docker が Mac 単機には重い。 | **本格案の後段** |
| [Plano（旧 Katanemo Arch）](https://github.com/katanemo/plano) | Apache-2.0。Envoy-based proxy。`planoai up`。既定の Plano-Orchestrator model は hosted（US-central）なので、完全 self-host には Ollama/vLLM の routing model 設定が必要。 | model name、alias、preference、domain/action、cost/latency catalog、ordered candidate fallback。 | `/v1/chat/completions`、`/v1/messages`、`/v1/responses`、state management を docs に記載。 | `route_on_user_only`／`X-Model-Affinity` を使わないと agentic loop 中に model が変わり得る。HTTP agent orchestration は Codex local child process と別物。 | semantic+gateway の実験候補。hosted dependency と affinity を受入条件にする。 |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Apache-2.0。learned strong/weak router framework。 | MF、weighted Elo、BERT、causal、threshold。 | 公式 server は OpenAI Chat Completions surface。tools/streaming 等の Chat fields は扱うが、Responses／Anthropic public endpoint の drop-in parity はない。 | Codex Responses provider の直接置換にならない。別 classifier として使うなら protocol gateway が必要。 | **研究／部品** |
| [Dispatch](https://github.com/OpusNano/dispatch) | MIT。OpenCode＋OpenRouter 専用の self-hosted Docker service。2026-08-23 時点で 0 stars／release なし。 | full conversation ではなく active task frame を抽出し、easy／medium／hard／critical の4段階へ deterministic classification。manual alias/header override と task-scoped escalation を持つ。 | `/v1/chat/completions` のみ。upstream は OpenRouter 固定。health／readiness／hot reload／metadata-only debug はあるが、Responses／Anthropic／provider pool はない。 | OpenCode の低コスト自動選択として設計は直接的。ただし Codex Responses、native child relay、OpenCode Go 直結を扱わないため、採用候補より classifier の設計資料。 | **小型の設計参考** |
| [aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router) | MIT。Python decision layer、local/HuggingFace 等の semantic vector。 | route name と embedding／hybrid decision。 | HTTP gateway、provider registry、stream/tool/fallback/health は提供しない。 | 自作 policy に組み込む小部品。置換製品ではない。 | **部品** |
| [OptiLLM](https://github.com/algorithmicsuperintelligence/optillm) | Apache-2.0。OpenAI-compatible optimizing inference proxy。 | optimization techniques、`router` plugin、ModernBERT による approach selection。proxy は health/round-robin/failover。 | OpenAI 互換中心。Responses、Anthropic Messages、Codex custom tool の parity は未確認。 | reasoning optimizer としての実験対象だが、現行 adapter を置き換える根拠は不足。 | **研究／部品** |
| [Not Diamond](https://docs.notdiamond.ai/docs/what-is-not-diamond) | hosted intelligent router。Python/TS/REST、account/API key。公式 Python repo は archive 済みで、license file も確認できない。 | prompt/task/model capability に基づく hosted selection。 | service endpoint。self-host／OSS router ではない。 | OpenRouter Auto の裏側を含め、credential・prompt が外部 service を通る。除外。 | **hosted／除外** |

### Hosted router を代替候補から外す理由

- **OpenRouter Auto** は hosted の auto-router／fallback であり、OpenAI-compatible endpoint の model selection を簡素化する。一方、provider credential、prompt、Codex native child session／encrypted relay を local boundary に置く OSS ではない。[Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router)・[model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks)
- **Requesty** は公式 quickstart が `https://router.requesty.ai/v1` と API key を要求し、OpenAI／Anthropic-compatible endpoint、policy fallback、load balancing、analytics を hosted gateway として提供する。provider registry／fallback の比較対象にはなるが、self-hosted Codex relay ではない。[quickstart](https://docs.requesty.ai/quickstart)・[fallback／routing](https://www.requesty.ai/)
- **Martian** は公式 Gateway docs が単一 API で 200+ model にアクセスする hosted gateway を説明し、SDK の Router は account／training job で quality-latency routing を作る。公開 `martianrouter` repo も account 作成を前提とするため、local one-port OSS gateway の代替には数えない。[Gateway docs](https://gateway-docs.withmartian.com/)・[SDK Router](https://withmartian.github.io/martian-sdk-python/api/router.html)・[martianrouter repo](https://github.com/martianprotocol/martianrouter)
- **Unify** は同名の model router と取り違えやすいが、現行の公式 `unifyai/unify` repo は hosted AI teammates／workspace integrations の product identity である。今回の model gateway 候補には入れない。[repo](https://github.com/unifyai/unify)

## 最有力候補の深掘り

### 1. `duolahypercho/codex-router`: 今回のスコープに最も近い Codex 専用 OSS

`duolahypercho/codex-router` は、一般的な LLM gateway ではなく、Codex App/CLI に external models を安全に載せる専用 router である。MIT license、macOS/Linux の per-user service、Homebrew、tray/control、Node.js 22.19+ と `uv`/Python の構成を公式 README に記載する。現行 main は親セッションで `e8c428d`（2026-08-23）を確認し、公式 release は `v0.4.0-beta.4`（2026-08-15、release commit `2376def`）。[README](https://github.com/duolahypercho/codex-router)・[MIT LICENSE](https://github.com/duolahypercho/codex-router/blob/main/LICENSE)・[releases](https://github.com/duolahypercho/codex-router/releases)

現在の自作スタックに対応する部品は次のとおり。

- **Codex 入口と catalog**: Responses API の local endpoint と caller capability を持ち、外部 model を Codex native catalog に mergeする。provider/model の選択、表示、credential 状態を router state の allowlist として管理する。[README の Codex integration](https://github.com/duolahypercho/codex-router#make-models-appear-in-codex)
- **OpenCode Go catalog**: `opencode-go`（Chat Completions）、`opencode-go-messages`（Anthropic Messages）、`opencode-go-responses`（Responses）を別 protocol variant として登録し、同じ Go/Zen provider family の key で選択できる。OpenCode Go の catalog には DeepSeek、Kimi、Grok、GLM、Qwen、GPT 5.6 Luna 等が並ぶが、一覧に載ることと native v2 child eligibility は別である。[OpenCode Go section](https://github.com/duolahypercho/codex-router#opencode-go-subscription-and-zen)
- **Codex native relay**: native tool namespace inventory を routed model に渡し、parent が spawned child を同じ model selection で扱える relay を持つ。`apply_patch` の custom/freeform shape と ordinary function shape の橋渡しも実装対象に含む。release history には「full native toolset and namespaces」「spawned threads の session model 継承」「routed agent delegation の hardening」「native tool／encrypted-payload relay」の変更が記録されている。[release history](https://github.com/duolahypercho/codex-router/releases)・[namespace relay](https://github.com/duolahypercho/codex-router/blob/e8c428d/src/namespace-relay.mjs)・[router source](https://github.com/duolahypercho/codex-router/blob/e8c428d/src/router.mjs)
- **native encrypted child-task relay**: v2 gate は、streamed Responses、forced function call、native Codex parent→child の encrypted payload relay、exact marker、同じ child への follow-up の 5 checks を route 単位で要求する。[How it works](https://github.com/duolahypercho/codex-router/blob/e8c428d/docs/HOW-IT-WORKS.md)・[v2 agent README](https://github.com/duolahypercho/codex-router/blob/main/v2_agent/README.md) これは通常の `/responses` smoke test や plaintext marker だけでは証明できないという設計である。
- **fallback／health／control**: 402、quota exhaustion、長い 429 の場合に enabled/credentialed model の chain を再構築し、provider の reset window を尊重する。`doctor`、`control failover status/chain/reset`、tray の実際に応答した model 表示、rollback ref、health endpoint がある。[failover section](https://github.com/duolahypercho/codex-router#keep-working-when-a-provider-runs-out-of-usage)
- **credential isolation**: Codex からは random local caller capability のみを見せ、provider key は forwarder で注入する。managed config は marker block に限定し、native OpenAI provider、profiles、MCP、trust、reasoning defaults を保つ。[architecture](https://github.com/duolahypercho/codex-router#how-routing-works)・[security](https://github.com/duolahypercho/codex-router/blob/main/SECURITY.md)

ここが CCR と決定的に違う。CCR は non-GPT model 用の `apply_patch` bridge は持つが、current main の `codex-multi-agent-bridge` は `multi_agent_v1` 固定であり、native encrypted child payload relay と同等の実装証拠は確認できない。`codex-router` は Codex child 自体を別 harness として作るのではなく、Codex の native spawn／namespace／encrypted payload を relay するので、今回の「Codex のフル機能を壊さない」という要件に最も近い。

ただし二つの留保がある。

1. **v2 は exact provider/model route の証明**。`v2_agent/README.md` は、model family ではなく route ごとに application と registry change を要求し、6 exact Kimi/Grok identity を grandfathered と記載する。親セッションで current config を確認した結果、DeepSeek と OpenCode Go は catalog／v1 route にはあるが v2 child route として認証済みではない。DeepSeek child／OpenCode Go child を採用するには、実 quota を使う 5 checks、redacted proof、`multiAgentVersion: "v2"` の upstream application が必要である。
2. **protocol adapter の完全な置換ではない**。README の architecture flow は `Codex Responses :4202 -> LiteLLM :4200 -> providers` であり、現行 LiteLLM＋`sitecustomize.py` と同じ責務が一部残る。`codex-router` を入れたから直ちに LiteLLM を削除できる、とは言わない。まず catalog／relay／control を upstream に寄せ、protocol translation は同じ replay fixture で判定する。

**判断**: current `codex-router` の model catalog、namespace relay、native encrypted child relay、failover/control が欲しいなら、これを第一 PoC とする。DeepSeek/OpenCode Go の exact child は v2 application と upstream change までを含む未完了作業であり、Kimi/Grok の grandfathered v2 と同じ扱いにしない。

### 2. CCR: 現スタックの「gateway 部分」に最も近い汎用候補

CCR の README は Claude Code だけでなく Codex、OpenCode、Grok CLI、Kimi CLI 等を同じ local control plane で扱うと説明している。MIT LICENSE で、macOS desktop DMG と CLI、既定 gateway `127.0.0.1:3456` がある。[README](https://github.com/musistudio/claude-code-router)・[MIT LICENSE](https://raw.githubusercontent.com/musistudio/claude-code-router/main/LICENSE)

対応する責務はかなり具体的である。

- provider は OpenAI Chat Completions、OpenAI Responses、Anthropic Messages、Gemini 等を設定できる。[provider guide](https://github.com/musistudio/claude-code-router/blob/main/docs/src/content/docs/en/guides/provider.md)
- route は request の header/body、model、messages、tools 等を見て書き換えられ、rule／global fallback と retry/failover がある。[routing](https://github.com/musistudio/claude-code-router/blob/main/docs/src/content/docs/en/configuration/routing.md)
- Codex の `apply_patch` custom/freeform tool は、non-GPT upstream へ ordinary function `virtual_apply_patch` として送り、upstream response を Codex custom tool call に戻す専用 adapter がある。CCR はファイルを直接変更せず、Codex が patch を実行する。[Codex routing docs](https://github.com/musistudio/claude-code-router/blob/main/docs/src/content/docs/en/configuration/routing.md)
- release history には Codex provider 404、Responses normalization、subagent model config handling の修正がある。これは Codex 経路を実装・保守している証拠だが、OpenCode Go 固有の互換性を保証するものではない。[releases](https://github.com/musistudio/claude-code-router/releases)

適用範囲は「LiteLLM＋`sitecustomize.py` を一気に消す」ではなく、まず CCR を同じ provider alias の shadow endpoint に置き、Responses replay を比較すること。特に `response.created`／`response.output_item.added`／`function_call`／`custom_tool_call`／stream 終了、reasoning metadata、未回答 tool call 除去、`/models`、OpenCode Go の error shape を確認する。

残るリスクは二つある。

1. CCR は Codex child agent を生成する harness ではない。Codex の full-capability child delegation は CCR の外で起きる。
2. Desktop の構成変更について、公式 issue #1647 は削除後にも共有 ChatGPT/Codex config が dead `127.0.0.1:3456` を指し得ると報告している。[issue #1647](https://github.com/musistudio/claude-code-router/issues/1647) したがって、global `~/.codex/config.toml` ではなく isolated config、backup、アンインストール後の health check を必須にする。

**判断**: Mac 単機で最短に protocol adapter を減らす候補は CCR。ただし「semantic router」と呼ぶ場合は、CCR の dynamic script（ユーザーが書く条件 router）と learned semantic model router を混同しない。

### 3. Bifrost: Go の provider/fallback gateway としての本命

Bifrost は core Apache-2.0 の Go gateway で、23+ provider、OpenAI-compatible API、failover/load balancing、streaming、multimodal、MCP を公式 README に記載している。[README](https://github.com/maximhq/bifrost)・[LICENSE](https://raw.githubusercontent.com/maximhq/bifrost/main/LICENSE)

今回の構成に対する直接性が高い。

- 公式 Codex guide は `http://localhost:8080/openai/v1`、`wire_api = "responses"`、non-OpenAI model では `supports_websockets = false` を示す。model 名を `provider/model-name` として Anthropic、Google、Mistral 等へ翻訳する。[Codex CLI guide](https://docs.getbifrost.ai/cli-agents/codex-cli)
- 公式 OpenCode guide は Bifrost の `/openai` または `/anthropic/v1` を base URL とし、provider-qualified model、thinking/reasoning options、tool calling を設定する。[OpenCode guide](https://docs.getbifrost.ai/cli-agents/opencode)
- CEL 条件で header/body/budget/token を見て route し、rule chain、A/B、capacity/budget/org scope を構成できる。[routing rules](https://docs.getbifrost.ai/providers/routing-rules)
- retries は network/5xx/429 と exponential backoff、key rotation。retries 後に sequential fallback を実行する。[retries/fallbacks](https://docs.getbifrost.ai/features/retries-and-fallbacks)
- model catalog は model→provider 解決、weighted／budget／rate-limit／performance metrics を扱う。adaptive load balancing は enterprise なので core と区別する。[provider routing](https://docs.getbifrost.ai/providers/provider-routing)

Bifrost で置き換えられるのは provider registry、protocol adapter、fallback/health、static routing であり、Codex の subagent spawn は残す。semantic complexity routing は CEL で task class のような明示的 signal を route する設計に留める。Bifrost core で GPT 系以外の Responses tool transcript をどこまで lossless に翻訳できるかは、CCR と同じく実測する。

**判断**: LiteLLM の構成を Go／単体プロセス寄りに整理したい場合の第二 PoC。Bifrost と CCR を同時に使って二重変換する必要はない。

### 4. OMO／LazyCodex: Codex-native な role policy だが gateway ではない

`oh-my-openagent` は旧 `oh-my-opencode` からの rename 中で、README は OpenCode Ultimate、Codex CLI Light、Senpi の三つの edition を説明している。Light edition は Codex plugin system、Codex agent TOML、`~/.codex/agents/` を使い、OpenCode の `team_*` surface ではなく Codex の own spawn/collaboration surface を使う。[README](https://github.com/code-yeongyu/oh-my-openagent)

agent model matching は二種類の fallback を区別する。

- `model-fallback`: agent/category の hardcoded requirement を満たす model chain を proactive に解決。
- `runtime-fallback`: session error 後に別 model へ移す reactive fallback。

この構造は現在の「per-model custom agent TOML＋親 AGENTS policy」を role/category の設定に寄せる用途には近い。[agent model matching guide](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/docs/guide/agent-model-matching.md)

ただし、core LICENSE は SUL-1.0 で、internal business／commercial use の条件があり、OSI open-source と分類できない。[LICENSE](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/LICENSE.md) `lazycodex-ai` の MIT は「oh-my-openagent を Codex にインストールする alias/distribution layer」の license である。[LazyCodex LICENSE](https://github.com/code-yeongyu/lazycodex/blob/main/LICENSE) 本体を strict OSS shortlist に入れる場合は、この差を明記する。

また、LazyCodex の公式 issue には `multi_agent_v2 = false` の blanket override が新しい Codex の `collaboration.spawn_agent` schema と衝突する事例がある。[issue #118](https://github.com/code-yeongyu/lazycodex/issues/118) model-aware な multi-agent version を無効にする設定をコピーせず、Codex version ごとに migration test を行う。

**判断**: gateway の置換ではなく、Codex 内の model/role selection の整理候補。license を受け入れられない場合は、同じ idea を既存の AGENTS policy と agent TOML に小さく実装する方が安全である。

### 5. Ruflo: OSS の agent meta-harness、ただし表面積が大きい

Ruflo は `ruvnet/ruflo` が canonical name で、README は「Claude Code and Codex の agent meta-harness」と説明する。MIT license、MCP/CLI、100+ specialized agents、swarm、memory、hooks、multi-provider routing を掲げる。[README](https://github.com/ruvnet/ruflo)・[MIT LICENSE](https://raw.githubusercontent.com/ruvnet/ruflo/main/LICENSE)

公式 docs には `hooks route`、`agent spawn`、agent role routing、local LLM／provider routing がある。[CLI reference](https://github.com/ruvnet/ruflo/wiki/CLI-Reference)・[agents](https://github.com/ruvnet/ruflo/wiki/Agents) README の概念図は `User -> Ruflo (CLI/MCP) -> Router -> Swarm -> Agents -> ... -> LLM Providers` と明記している。

ただし次の理由で、今回の gateway replacement にはしない。

- 公式導線は Claude Code plugin/MCP が中心で、Codex custom provider の Responses wire adapter としての保証は見当たらない。
- 98 agents、60+ commands、30+ skills、MCP server、hooks、daemon の導入は、現在の Codex harness を維持する「最小変更」の反対方向である。[install/paths](https://github.com/ruvnet/ruflo/wiki/Installation)
- README の「89% accuracy」「5 providers with failover」「self-learning」は製品側の主張であり、今回の OpenCode Go／Codex tool replay に対する独立証拠ではない。
- current main、wiki、npm の version 表示が同時点で一致しない場合があるため、PoC では `npx` の floating latest ではなく commit／tag を pin する。

**判断**: Codex を含む複数 harness の大規模 swarm を試すなら候補。ただし今回の「router の自作を減らす」目的には過剰で、まず CCR/Bifrost＋Codex native spawn を評価する。

### 6. Semantic router: vLLM SR と Plano の使い分け

#### vLLM Semantic Router

vLLM SR は Apache-2.0 の decision layer で、Envoy extproc と YAML policy の間に位置する。router は model weights を load せず、OpenAI-compatible backend を選ぶ。[overview](https://vllm-sr.ai/docs/overview/semantic-router-overview/)・[installation](https://vllm-sr.ai/docs/installation/)

v0.3 router API は public `/v1/chat/completions` と `/v1/responses` を持ち、Responses は内部で Chat Completions へ翻訳する。backend として OpenAI-compatible Chat と Anthropic Messages を扱い、Anthropic conversion は tools/tool_calls/tool messages と streaming SSE tool deltas を含む。[router API](https://vllm-sr.ai/docs/v0.3/api/router/)

重要な制約は、公式 fallback proposal が「cross-model fallback は将来、現状は同一 backend cluster の retry/outlier ejection が中心」としていること。LiteLLM/Bifrost の ordered cross-provider fallback を semantic router だけで置き換えない。[fallback proposal](https://vllm-sr.ai/docs/proposals/model-execution-fallback/)

#### Plano（旧 Katanemo Arch）

Plano は Apache-2.0 の Envoy-based AI proxy。`Arch` repo は `Plano` に移転／redirect されている。[Plano repo](https://github.com/katanemo/plano)・[Arch redirect](https://github.com/katanemo/arch)

LLM routing docs は model name、alias、preference-aligned Plano-Orchestrator、cost/latency、ordered candidates（先頭 primary、後続 fallback on 429/5xx）を記載する。[LLM routing](https://docs.planoai.dev/guides/llm_router.html)

provider docs は `/v1/chat/completions`、`/v1/messages`、`/v1/responses` と OpenAI/Anthropic/Azure/DeepSeek/OpenAI-compatible adapters を記載し、state docs は Responses の `resp_id` と prior context を管理する。[providers](https://docs.planoai.dev/concepts/llm_providers/supported_providers.html)・[state](https://docs.planoai.dev/guides/state.html)

一方、docs は default で request ごとに routing すると agentic loop の途中で model が変わる可能性を警告する。`route_on_user_only = true` または `X-Model-Affinity` を有効にして user turn 単位で pin する必要がある。[affinity](https://docs.planoai.dev/guides/llm_router.html)

**判断**: semantic model selection を研究する本格案にはなるが、まず protocol fidelity を CCR/Bifrost で固定してから追加する。vLLM SR と Plano を同時に入れず、一方を decision layer、もう一方を使わない。tool loop 中の model switch は不合格とする。

## 置換マトリクス

記号: ◎ = 公式資料が直接対応、○ = 部分的／構成次第、△ = 別実装が必要、— = 主責務外。

| 候補 | protocol adapter | provider registry | static policy | semantic／learned routing | fallback／health | Codex child spawn／native relay | current stack の置換 |
|---|---:|---:|---:|---:|---:|---:|---|
| duolahypercho/codex-router | ○ Responses ingress。README の current flow は内部 LiteLLM adapter | ◎ catalog／credential-aware picker | ◎ provider/model selection、v2 eligibility | —（semantic classifier ではない） | ◎ quota-aware failover、health、doctor、rollback | ◎ native namespace／encrypted child-task relay。ただし exact route ごとに v2 proof | relay/catalog/control を upstreamize。LiteLLM を消せるかは別 PoC |
| CCR | ◎ Responses/Chat/Anthropic＋Codex patch bridge | ◎ | ◎ body/header/script/model-chain | △ user script。learned router ではない | ◎ | △ Codex を保持するが native encrypted relay は確認できない | LiteLLM＋patch の大部分。ただし PoC 必須 |
| Bifrost | ◎ Responses ingress、OpenAI/Anthropic等への adapter | ◎ catalog | ◎ CEL | — | ◎ | —（Codex を保持） | LiteLLM＋patch の大部分。Go gateway 候補 |
| LiteLLM | ◎ Responses/Chat/Messages | ◎ | ◎ weighted/cost/latency等 | △ custom strategy | ◎ | —（Codex を保持） | 現在の基準線 |
| OMO/LazyCodex | — | △ agent profile | ◎ agent/category chain | △ category routing。main prompt は別途確認 | ◎ model/runtime fallback | ◎ Codex native surface を利用 | agent TOML／role policy のみ |
| Ruflo | △ wire adapter は未確定 | ○ multi-provider claim | ◎ hooks/agent routing | ○ official claim。実測要 | ○ failover claim | △ external MCP/swarm。Codex native semantics は要検証 | harness/policy の大幅変更 |
| CodeRouter | △ provider adapter／MCP | ○ per-provider config | ◎ task/workflow routing | ○ cognitive-shape＋run history | ○ validators／cost ceiling | △ Codex から別 MCP agent を呼ぶ | 新興 meta-agent。native relay の置換ではない |
| OpenCode | ○ provider SDK（Chat/Responses） | ○ per-provider config | ◎ agent config | — | △ gateway fallback ではない | — OpenCode 自身の child session | Codex harness の置換 |
| vLLM SR | ○ Responses→Chat、Anthropic backend | ○ backend refs | ◎ YAML/signals | ◎ | △ cross-model fallback は未実装方向 | — | semantic layer の追加 |
| Plano | ◎ Chat/Messages/Responses/state | ◎ provider adapters | ◎ alias/preference/cost/latency | ◎ Plano-Orchestrator | ◎ ordered 429/5xx | — HTTP agent orchestration | semantic+gateway の本格案 |
| RouteLLM | △ OpenAI Chat server | — | ○ threshold | ◎ learned strong/weak | — | — | classifier 部品 |
| Dispatch | △ Chat Completions／OpenRouter 固定 | — | ◎ alias/header/4 tier | ○ active-task-frame heuristic | ○ local health/readiness | — OpenCode 専用 | 小型 classifier の設計参考 |
| Portkey | ○ gateway adapters | ◎ | ◎ conditional/fallback | — | ◎ | — | provider/fallback の代替候補 |
| Helicone | ○ OpenAI-compatible | ◎ managed/BYOK | ◎ provider selection | △ fastest/cheapest/reliable | ◎ | — | license／cloud 境界のため保留 |
| TensorZero | ◎ | ◎ | ◎ | ○ | ◎ | — | archived のため不可 |
| Envoy/Higress/APISIX/Kong | ○ provider/proxy 範囲 | ◎ | ◎ | △ 製品／plugin 次第 | ◎ | — | Mac 単機には過大 |

特に重要なのは最後から二列目である。**gateway の ◎ は Codex child spawn の ◎ ではない**。CCR/Bifrost/LiteLLM を入れても Codex が full-capability child agent を呼べる条件は別に満たす必要がある。`codex-router` は例外的に Codex native child payload の relay を持つが、それでも child session 自体を生成するのは Codex である。

## 明確な不適合理由

### 「agent 製品」を gateway の代わりにする誤り

OpenCode、Roo、Continue、Aider、OpenHands、Ruflo は、agent の tool loop、mode、session、workflow、MCP、swarm を提供する。これらを入れると model policy は簡素化できることがあるが、Codex の harness を保つという前提を破る可能性がある。OpenCode には build/plan/general subagent があるが、それは OpenCode の session model である。[OpenCode agents](https://opencode.ai/docs/agents/)

### 「gateway」を agent spawn と誤解すること

CCR、Bifrost、LiteLLM、Portkey、Helicone は HTTP gateway であり、Codex の child session を作らない。`codex-router` は例外的に native child payload／namespace を relay するが、child session の spawn、filesystem、tools、permissions は引き続き Codex 側の責務である。gateway upstream が function/tool call を返すだけでは full-capability child の証明にならない。

### Semantic router を tool loop に直結すること

task classifier が各 request を独立に分類すると、tool result を受け取った次の turn で別 model へ移る。これは stateful reasoning、tool schema、prompt cache、provider-specific IDs を壊す。Plano docs 自身が model affinity を要求し、vLLM SR も Responses↔Chat の変換を行う。semantic router は「新しい user turn の decision」に限定し、stream 開始後は gateway の retry/fallback を含めても model を変えない。

### license／継続性を OSS と混同すること

- OMO 本体は SUL-1.0、LazyCodex の MIT は installer。
- Helicone `ai-gateway` は current raw LICENSE が GPLv3。旧 repo／badge の Apache 表記と同一視しない。
- Roo は Apache-2.0 でも repo archive／extension shutdown 済み。
- Continue は Apache-2.0 でも公式に read-only/final release。
- TensorZero は Apache-2.0 でも owner archive 済み。

### Kubernetes gateway を Mac の local router と誤認すること

Envoy AI Gateway、Higress、APISIX、Kong、Plano、vLLM SR は production data plane としては有力だが、Envoy/Kubernetes/Docker/etcd 等を伴う。単機 macOS で port 1 個の local gateway を置き換える目的には、Codex 専用の `codex-router`、または CCR/Bifrost/LiteLLM の方が実験面積が小さい。

## 推奨アーキテクチャ

### A. 最小変更案（推奨）

```text
Codex
  ├─ AGENTS policy + agent TOML（Luna -> DeepSeek -> Terra -> Sol）
  ├─ tools / filesystem / MCP / child delegation
  └─ Responses custom provider
       -> codex-router（catalog／native relay／control）
            -> 現行 LiteLLM または provider adapter
                 -> OpenCode Go (Chat/Anthropic-compatible upstream)
```

手順は次のとおり。

1. `codex-router` を別 port／isolated config で起動し、現在の LiteLLM endpoint と同じ logical model alias を一つだけ shadow 移行する。
2. まず catalog、provider credential injection、namespace relay、`apply_patch` bridge、health／failover／control を確認する。Codex の native spawn と filesystem/tools/permissions はそのまま残す。
3. v2 grandfathered route か、5 checks を通して application 済みの exact route だけで encrypted child relay を検証する。DeepSeek／OpenCode Go は v2 application 前は v1 扱いに固定する。
4. direct Responses replay、streaming、reasoning、function call、Codex `apply_patch`、child delegation を現行経路と比較する。LiteLLM／`sitecustomize.py` はこの段階では外さない。
5. 全 acceptance criteria を通した後に、protocol adapter を一段ずつ CCR または Bifrost の shadow endpoint と比較する。失敗時は provider URL の切替だけで戻せるようにする。
6. semantic routing は入れず、AGENTS policy が明示的に `fast`／`deep`／`ultra` の logical alias を選ぶ。

`codex-router` は Codex native relay／exact-route v2 gate を優先する場合の第一候補、CCR は LiteLLM を軽い local gateway に置き換えられるかを測る次点、Bifrost は Go gateway／Codex/OpenCode の両方を一つの provider plane に寄せたい場合の候補である。現時点でどれが OpenCode Go の全モデルを無修正で受けるかは、ドキュメントだけでは決められない。

### B. 本格案（semantic routing を追加する場合）

```text
Codex user turn
  -> semantic decision（vLLM SR または Plano、user-turn boundary のみ）
  -> logical model alias
  -> CCR または Bifrost（protocol adapter + provider + reliability）
  -> provider pool / OpenCode Go

Codex tool loop ------------------------------+
  model affinity / route_on_user_only  <------+
```

実装上は二つの選択肢がある。

- vLLM SR／Plano が Responses ingress を持つため、そこを最初の入口にする。ただし内部 conversion を一度増やすので、response ID、reasoning、tool call、stream event の lossless replay を証明する。
- semantic classifier を preflight service として切り出し、分類結果だけを `X-Task-Class` または logical alias にして CCR/Bifrost に渡す。こちらは Codex の wire path を一つに保ちやすく、現スタックに対する差分が小さい。

後者を推奨する。learned semantic router が「もっと賢そう」に見えても、Codex の tool protocol を二重に変換する価値は acceptance test で測るまで仮定しない。

## PoC acceptance criteria

以下は製品の公式機能表ではなく、今回の採否を決めるための提案条件である。

### Protocol／tool fidelity

1. `POST /v1/responses` の non-streaming と streaming を通す。
2. `GET /v1/models`、model alias、OpenCode Go の upstream `/models` path を通す。
3. plain text、reasoning metadata、single function call、複数 tool call、tool result、長い multi-turn replay を通す。
4. Codex `apply_patch` custom/freeform tool の往復を通し、patch は Codex が実行することを確認する。未回答 tool call、tool adjacency、duplicate response item、壊れた `previous_response_id` をゼロにする。
5. Chat Completions と Anthropic Messages の両方を provider adapter のテスト fixture に含める。OpenCode Go が一方しか受けない場合に、gateway が正しい protocol を選べることを確認する。

### Agent／routing semantics

6. `codex-router` の exact provider/model route を採用する場合、v2 application／registry eligibility を先に確認する。streamed Responses、forced function call、native encrypted parent→child relay、exact marker、同じ child への follow-up の 5 checks が揃わない route は catalog に表示されても v2 child として合格にしない。
7. Codex の full-capability parent が filesystem、MCP、shell、tool を使用できる。parent が child agent を生成し、child が別 logical model alias を使い、child completion を parent が受け取る。nested child が必要なら同じ fixture を追加する。
8. gateway／semantic layer は Codex の tool permission、child session ID、working directory、environment を変更しない。
9. 429、5xx、connect refused、timeout、provider health failure を synthetic upstream で再現する。retry／fallback は response bytes が始まる前だけ行い、stream 開始後は同じ request を二重実行しない。
10. semantic routing を試す場合、同じ tool loop 中に model/provider が変わらない。reroute は user turn boundary のみ。classifier の allowlist、unknown/error の deterministic fallback を定義する。

### Local operations／safety

11. macOS で start/stop/restart、port conflict、`/health`／readiness、ログの secret redaction、process count、RSS、cold start を記録する。Docker が必要なら「単機軽量」要件からの逸脱を明示する。
12. CCR の global Codex config mutation を検出し、isolated config と backup/restore を通す。[CCR issue #1647](https://github.com/musistudio/claude-code-router/issues/1647)
13. upstream に送られた request と Codex の original transcript を sanitized fixture で diff し、rollback は provider URL 一つの変更で行える。
14. source commit/tag、package lock、license を pin する。Ruflo のように README/wiki/npm の version がずれる候補は `latest` を本番に使わない。

### 採用ゲート

- protocol fixture は **100% pass**。一つでも orphan tool call、tool schema loss、stream event loss があれば現行 LiteLLM を外さない。
- parent/child Codex fixture は full-capability で pass。subagent が呼べない場合、gateway を交換しても解決したとはみなさない。
- fallback は fault injection の全ケースで duplicate side effect がゼロ。
- semantic router は offline labeled set で採点するが、品質が上がっても tool-loop pin が破れるなら不採用。
- license、secret locality、macOS の実測値を記録し、未検証の公式 claim を採用理由にしない。

## 調査の限界と次の一手

本稿は公式資料の landscape 調査であり、OpenCode Go に対する `codex-router`／CCR／Bifrost の live compatibility、DeepSeek／OpenCode Go exact route の v2 child、macOS RSS、tool-loop の exact transcript はまだ測っていない。したがって「第一候補」は機能表と親セッション観測からの優先順位であって、導入完了の宣言ではない。

最初に実行する順番は **`codex-router` shadow PoC → CCR shadow PoC → Bifrost shadow PoC → gateway／relay の採用判断 → その後に semantic classifier**。OMO/LazyCodex と Ruflo は、gateway の採否とは別に Codex agent policy／orchestration の候補として評価する。strict OSS が条件なら OMO 本体は候補から外し、既存 AGENTS/TOML を整理する。

### 主要一次資料（カテゴリ別）

- Codex Router: [repo](https://github.com/duolahypercho/codex-router)、[LICENSE](https://github.com/duolahypercho/codex-router/blob/main/LICENSE)、[How it works](https://github.com/duolahypercho/codex-router/blob/e8c428d/docs/HOW-IT-WORKS.md)、[v2 agent gate](https://github.com/duolahypercho/codex-router/blob/main/v2_agent/README.md)、[namespace relay](https://github.com/duolahypercho/codex-router/blob/e8c428d/src/namespace-relay.mjs)、[router source](https://github.com/duolahypercho/codex-router/blob/e8c428d/src/router.mjs)、[OpenCode catalog research](https://github.com/duolahypercho/codex-router/blob/e8c428d/docs/research/opencode-catalog-2026-08-21.md)、[releases](https://github.com/duolahypercho/codex-router/releases)、[security](https://github.com/duolahypercho/codex-router/blob/main/SECURITY.md)
- OpenAI Codex upstream: [rust-v0.148.0 role implementation](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/agent/role.rs)、[role tests](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/agent/role_tests.rs)、[v2 provider override test](https://github.com/openai/codex/blob/rust-v0.148.0/codex-rs/core/src/tools/handlers/multi_agents_tests.rs#L902-L956)、[release](https://github.com/openai/codex/releases/tag/rust-v0.148.0)
- CCR: [repo](https://github.com/musistudio/claude-code-router)、[routing](https://github.com/musistudio/claude-code-router/blob/main/docs/src/content/docs/en/configuration/routing.md)、[provider](https://github.com/musistudio/claude-code-router/blob/main/docs/src/content/docs/en/guides/provider.md)、[releases](https://github.com/musistudio/claude-code-router/releases)、[issue #1647](https://github.com/musistudio/claude-code-router/issues/1647)
- Bifrost: [repo](https://github.com/maximhq/bifrost)、[Codex](https://docs.getbifrost.ai/cli-agents/codex-cli)、[OpenCode](https://docs.getbifrost.ai/cli-agents/opencode)、[routing rules](https://docs.getbifrost.ai/providers/routing-rules)、[fallback](https://docs.getbifrost.ai/features/retries-and-fallbacks)、[releases](https://github.com/maximhq/bifrost/releases)
- OMO/LazyCodex: [OMO repo](https://github.com/code-yeongyu/oh-my-openagent)、[OMO license](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/LICENSE.md)、[model matching](https://github.com/code-yeongyu/oh-my-openagent/blob/dev/docs/guide/agent-model-matching.md)、[LazyCodex](https://github.com/code-yeongyu/lazycodex)、[LazyCodex issue #118](https://github.com/code-yeongyu/lazycodex/issues/118)
- Ruflo: [repo](https://github.com/ruvnet/ruflo)、[license](https://raw.githubusercontent.com/ruvnet/ruflo/main/LICENSE)、[CLI reference](https://github.com/ruvnet/ruflo/wiki/CLI-Reference)、[agents](https://github.com/ruvnet/ruflo/wiki/Agents)
- OpenCode／other agents: [OpenCode repo](https://github.com/anomalyco/opencode)、[providers](https://opencode.ai/docs/providers/)、[agents](https://opencode.ai/docs/agents/)、[CodeRouter](https://github.com/Code-Router/CodeRouter)、[Roo](https://github.com/RooCodeInc/Roo-Code)、[Continue](https://github.com/continuedev/continue)、[Aider](https://github.com/Aider-AI/aider)、[OpenHands](https://github.com/OpenHands/OpenHands)
- Gateways: [LiteLLM](https://github.com/BerriAI/litellm)、[LiteLLM routing](https://docs.litellm.ai/docs/routing)、[LiteLLM reliability](https://docs.litellm.ai/docs/proxy/reliability)、[Portkey](https://github.com/Portkey-AI/gateway)、[Helicone AI Gateway](https://github.com/Helicone/ai-gateway)、[Helicone LICENSE](https://raw.githubusercontent.com/Helicone/ai-gateway/main/LICENSE)、[TensorZero](https://github.com/tensorzero/tensorzero)、[Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway)、[Higress](https://github.com/higress-group/higress)、[APISIX AI Gateway](https://apisix.apache.org/ai-gateway/)、[Kong AI Gateway](https://docs.konghq.com/gateway/latest/get-started/ai-gateway/)
- Semantic routers: [vLLM SR repo](https://github.com/vllm-project/semantic-router)、[vLLM overview](https://vllm-sr.ai/docs/overview/semantic-router-overview/)、[vLLM API](https://vllm-sr.ai/docs/v0.3/api/router/)、[vLLM fallback proposal](https://vllm-sr.ai/docs/proposals/model-execution-fallback/)、[Plano repo](https://github.com/katanemo/plano)、[Plano routing](https://docs.planoai.dev/guides/llm_router.html)、[Plano state](https://docs.planoai.dev/guides/state.html)、[RouteLLM](https://github.com/lm-sys/RouteLLM)、[Dispatch](https://github.com/OpusNano/dispatch)、[semantic-router](https://github.com/aurelio-labs/semantic-router)、[OptiLLM](https://github.com/algorithmicsuperintelligence/optillm)、[Not Diamond docs](https://docs.notdiamond.ai/docs/what-is-not-diamond)、[Not Diamond Python](https://github.com/Not-Diamond/notdiamond-python)
- Hosted comparison: [OpenRouter Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router)、[OpenRouter fallback](https://openrouter.ai/docs/guides/routing/model-fallbacks)、[Requesty quickstart](https://docs.requesty.ai/quickstart)、[Requesty](https://www.requesty.ai/)、[Martian Gateway](https://gateway-docs.withmartian.com/)、[Martian SDK Router](https://withmartian.github.io/martian-sdk-python/api/router.html)、[martianrouter](https://github.com/martianprotocol/martianrouter)、[Unify](https://github.com/unifyai/unify)
