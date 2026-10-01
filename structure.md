# LLMServingSim 2.0 코드 구조 분석

> 목적: LLMServingSim 2.0이 어떻게 구성되어 있고, 한 iteration이 어떤 경로로 시뮬레이션되는지 정리한다.
> 이후 "near tier(NPU 로컬 HBM)가 담당하던 연산/데이터를 HBF(High Bandwidth Flash)로 옮기는" 개조 작업의 기준 문서로 쓰는 것이 목표다.
>
> - 분석 대상: `https://github.com/casys-kaist/LLMServingSim` `main` @ `a4053bc` (2026/08)
> - 서브모듈: `astra-sim` @ `d346994` (casys-kaist fork), 그 안의 `chakra` @ `30221ab`, analytical memory backend @ `62313ef`, analytical network backend @ `5dc0232`
> - 코드 인용은 `파일:라인` 형식이며 위 커밋 기준이다.
> - 이 문서는 **코드 기준**으로 작성했다. 공식 문서(`docs/`)와 실제 코드가 다른 부분은 [§9](#9-문서와-코드가-다른-부분--주의사항)에 따로 모았다.

---

## 목차

1. [한눈에 보기](#1-한눈에-보기)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [End-to-end 실행 흐름](#3-end-to-end-실행-흐름)
4. [Python 프론트엔드 모듈별 상세 (`serving/`)](#4-python-프론트엔드-모듈별-상세-serving)
5. [Trace → Chakra 그래프 변환](#5-trace--chakra-그래프-변환)
6. [ASTRA-Sim 백엔드 (C++)](#6-astra-sim-백엔드-c)
7. [입력/설정 파일 포맷](#7-입력설정-파일-포맷)
8. [메모리 계층 모델 정리 (HBF 개조 관점)](#8-메모리-계층-모델-정리-hbf-개조-관점)
9. [문서와 코드가 다른 부분 / 주의사항](#9-문서와-코드가-다른-부분--주의사항)
10. [HBF 개조 시 수정 지점 지도](#10-hbf-개조-시-수정-지점-지도)
11. [부록: 주요 함수 인덱스](#11-부록-주요-함수-인덱스)

---

## 1. 한눈에 보기

LLMServingSim 2.0은 **"Python 프론트엔드가 vLLM 스케줄러를 흉내 내며 매 iteration의 실행 trace를 만들고, C++ ASTRA-Sim이 그 trace를 실행해 시간을 돌려주는"** 구조의 co-simulator다.

```
            ┌────────────────────────── Python frontend (python -m serving) ──────────────────────────┐
            │                                                                                          │
 dataset ──►│ Router ──► Scheduler(per instance) ──► Batch ──► trace_generator ──► graph_generator ────┼──► .et 파일
 (.jsonl)   │   ▲          │  (vLLM V1 continuous      │        (profiled latency     (Chakra LLMConverter│     (Chakra protobuf)
            │   │          │   batching, KV block pool)│         lookup + 텐서 크기)    in-process)         │        │
            │   │          ▼                           │                                                   │        │
            │   │     MemoryModel / TieredKVCacheManager / BlockPool (NPU, CPU, CXL tier)                 │        │
            │   │                                                                                       │        ▼
            │   └──── add_done(완료 처리) ◄── Controller (stdin/stdout IPC) ◄───────────────────────────────┼── ASTRA-Sim (C++)
            │                                                                                          │   - Workload: ET 노드 실행
            └──────────────────────────────────────────────────────────────────────────────────────────┘   - Sys: comp/mem/comm 스케줄
                                                                                                            - memory backend (analytical)
 profiler/perf/<hw>/<model>/<variant>/tp<N>/*.csv  ──(레이어별 실측 latency)──► trace_generator             - network backend (analytical)
 configs/cluster/*.json ──► config_builder ──► network.yml / system.json / memory_expansion.json ──────────► ASTRA-Sim
```

핵심 성질:

| 항목 | 내용 |
| --- | --- |
| 시간 단위 | 1 cycle = 1 ns (`FREQ = 1_000_000_000`, `serving/__main__.py:599`) |
| 연산 시간의 출처 | **실제 GPU(vLLM)에서 레이어별로 프로파일링한 latency CSV**를 보간해서 사용 (`trace_generator.py`). 연산 시간을 FLOPs/BW로 계산하지 않는다 (PIM만 예외적으로 analytical 모델). |
| 통신 시간의 출처 | ASTRA-Sim analytical network backend (`link_bw`, `link_latency`) |
| 메모리 전송 시간의 출처 | ASTRA-Sim analytical memory backend (`memory_expansion.json`의 `mem-bw`, `mem-latency`) |
| 스케줄러 | vLLM v0.19.0 V1 scheduler/KV block pool의 포팅 (prefix caching, chunked prefill, preemption) |
| 병렬화 | TP / PP / EP / DP(+ P/D disaggregation) 조합 |
| 결정성 | 완전 결정적. `serving/validate.sh`가 baseline clock과 "정확히 같음"으로 회귀 검사 |

> **HBF 개조 관점에서 가장 중요한 사실**: 현재 시뮬레이터에서 near tier(HBM) 접근 시간은 **별도 이벤트로 존재하지 않고, 프로파일된 레이어 latency(`comp_time`) 안에 암묵적으로 녹아 있다.** ASTRA-Sim은 이 값을 고정 지연으로 그대로 재생한다. 메모리 전송 노드(→ ASTRA-Sim memory backend가 시간 계산)가 따로 생기는 경우는 다음뿐이다.
> - weight를 NPU 밖(`cpu`, `cxl:N`)에 둔 레이어의 weight 로드
> - `--enable-local-offloading` 시 모든 weight 레이어의 로드
> - 첫 레이어 입력 로드 / 마지막 레이어 출력 저장 (`REMOTE`)
> - 하위 tier KV recall (`kv_load` row)
> - PIM attention (`PIM_COMP` 노드)
>
> 자세한 내용은 [§8](#8-메모리-계층-모델-정리-hbf-개조-관점).

---

## 2. 디렉토리 구조

HBF 개조와 관련 있는 부분 위주로 정리했다.

```
LLMServingSim/
├── serving/                       # 시뮬레이터 본체 (python -m serving)
│   ├── __main__.py                # CLI 파싱, ASTRA-Sim 서브프로세스 실행, 메인 이벤트 루프 (1272줄)
│   ├── run.sh                     # 기능별 예제 실행 메뉴
│   ├── validate.sh                # 회귀 검사 (모든 시나리오의 Total clocks 비교)
│   ├── validate-baselines.txt
│   └── core/
│       ├── config_builder.py      # cluster config → ASTRA-Sim 입력 3종 + placement 해석 (871줄)
│       ├── router.py              # 요청 로딩/라우팅, agentic session 의존성 (327줄)
│       ├── scheduler.py           # vLLM V1 스타일 스케줄러 (576줄)
│       ├── request.py             # Request / Batch 데이터 클래스
│       ├── memory_model.py        # weight/KV 크기 계산, tier별 pool 생성, calculate_sizes() (545줄)
│       ├── block_pool.py          # tier 하나의 KV block pool + prefix-cache index, Device enum
│       ├── kv_cache_manager.py    # 다계층 KV cache manager (NPU + victim tier)
│       ├── trace_generator.py     # perf DB 로드/보간, 레이어별 trace row 생성 (1982줄, 가장 큼)
│       ├── graph_generator.py     # trace rows → Chakra .et (in-process 변환 + 캐시)
│       ├── controller.py          # ASTRA-Sim stdin/stdout 프로토콜
│       ├── pim_model.py           # PIM 디바이스 analytical latency 모델
│       ├── gate_function.py       # MoE expert routing
│       ├── power_model.py         # 전력/에너지
│       ├── run_paths.py, utils.py, logger.py
├── configs/
│   ├── cluster/*.json             # 클러스터 토폴로지/하드웨어/메모리/placement
│   ├── model/<org>/<name>.json    # HF config.json 부분집합
│   └── pim/*.ini                  # PIM(DRAMSim3 스타일) 디바이스 파라미터
├── profiler/
│   ├── models/<model_type>.yaml   # 아키텍처 catalog + 레이어 sequence (시뮬레이터가 이 순서로 trace를 만든다)
│   ├── perf/<hw>/<model>/<variant>/tp<N>/{dense,per_sequence,attention,moe,skew,skew_fit}.csv + meta.yaml
│   └── core/ ...                  # vLLM 기반 레이어별 프로파일러 (GPU 필요)
├── workloads/*.jsonl              # 요청 trace (flat 또는 agentic session)
├── bench/                         # 실제 vLLM 실행 결과와 시뮬레이터 비교
├── scripts/compile.sh             # chakra pip install + ASTRA-Sim 빌드
└── astra-sim/                     # 서브모듈 (C++)
    ├── astra-sim/                 # 코어: system/, workload/, ...
    ├── extern/graph_frontend/chakra/src/converter/llm_converter.py   # trace → ET 변환기
    ├── extern/memory_backend/analytical/                             # 메모리 백엔드
    ├── extern/network_backend/analytical/                            # 네트워크 백엔드
    └── inputs/
        ├── system/system.json     # 템플릿 (run마다 복사 후 수정)
        └── runs/<run_id>/         # 실행마다 생성되는 network/system/memory/workload/trace
```

**작업 디렉토리 주의**: `serving/__main__.py:255-257`에서 `os.chdir("astra-sim")`을 하므로 런타임의 모든 상대경로는 `astra-sim/` 기준이다. 그래서 코드 곳곳에 `../configs/...`, `../profiler/...`가 등장한다.

---

## 3. End-to-end 실행 흐름

### 3.1 시작 단계 (`serving/__main__.py:main`)

| 순서 | 위치 | 하는 일 |
| --- | --- | --- |
| 1 | `__main__.py:255-257` | cwd를 `astra-sim/`으로 변경 |
| 2 | `__main__.py:260-375` | CLI 파싱 (`--cluster-config`, `--max-num-seqs`, `--max-num-batched-tokens`, `--prefix-storage`, `--enable-attn-offloading`, `--enable-local-offloading` …) |
| 3 | `__main__.py:406-408` → `config_builder.build_cluster_config()` | cluster config 해석, 병렬도 추론, NPU id 매핑, placement 해석, **ASTRA-Sim 입력 파일 3종 생성** (`network.yml`, `system.json`, `memory_expansion.json`) |
| 4 | `__main__.py:204-248` | 인스턴스별 runtime config (dtype, block size, prefix caching 등; 인스턴스 JSON 값이 CLI 기본값을 override) |
| 5 | `__main__.py:448-512` | `--enable-prefix-sharing` + `--prefix-storage CPU/CXL`이면 공유 prefix pool 생성 (CPU: 노드당 1개, CXL: 전체 1개) |
| 6 | `__main__.py:514-543` | 인스턴스마다 `Scheduler` 생성 → 내부에서 `MemoryModel` → `BlockPool`/`TieredKVCacheManager` 생성 |
| 7 | `__main__.py:565-580` | `Controller`, `Router`, (옵션) `PowerModel` 생성, dataset 로드 |
| 8 | `__main__.py:611-621` | 첫 요청 도착 시점까지 대기하는 1-레이어 "event" trace 생성 후 그래프로 변환 |
| 9 | `__main__.py:623-630` | ASTRA-Sim 바이너리(`build/astra_analytical/.../AnalyticalAstra`)를 `--workload/--system/--network/--memory-configuration` 인자로 `subprocess.Popen` (stdin/stdout 파이프) |

### 3.2 메인 루프 (`__main__.py:662-1159`)

ASTRA-Sim이 "어떤 NPU(sys)가 iteration을 끝냈고 지금 시각이 몇 cycle인지"를 stdout으로 알려주면, Python이 그 NPU에게 다음 workload를 stdin으로 알려주는 **핑퐁 구조**다.

```
while True:
    out = controller.read_wait(p)                    # "Waiting"이 나올 때까지 stdout 읽기
    sys, id, current = parse("sys[X] iteration N finished, C cycles, exposed communication E cycles.")
    router.route_arrived_requests(current)           # 도착한 요청을 인스턴스로 라우팅
    instance_id = npu2inst_mapping[sys]
    scheduler.add_done(id, sys, current)             # 끝난 batch 완료 처리 (토큰 생성, TTFT/ITL 기록, KV 해제)
    new_req = scheduler.schedule(current, sys, id)   # 다음 batch 구성 (start NPU만 새 batch를 만든다)
    if new batch (built here):
        trace = generate_trace(batch, ...)           # 레이어별 row 생성
        generate_graph(batch, ..., trace=trace)      # Chakra .et 파일 생성
        controller.write_flush(p, workload_path)     # ASTRA-Sim에 ".../workload/<name>/llm" 전달
    elif 다른 NPU가 기존 batch에 합류:
        controller.write_flush(p, 기존 batch의 workload_path)
    elif 할 일 없음:
        controller.write_flush(p, "pass" | "pass <next_arrival_ns>" | "pass -1")
    if 모든 요청 완료: "done" (해당 인스턴스 sleep) / 전체 완료면 "exit"
```

- Python → ASTRA-Sim 메시지: workload 경로, `pass`, `pass <ns>`(다음 도착 시각까지 재질의 억제), `pass -1`(상태가 바뀐 pass; 모든 NPU 재질의 허용), `done`, `exit` (`__main__.py:65-89`, `1142-1156`).
- `add_done`의 `id -= 1` (`scheduler.py:362`): ASTRA-Sim이 보고하는 iteration 번호 - 1 == `batch_id`.
- 한 instance가 여러 NPU를 쓰면(TP/PP) **start NPU만 batch를 새로 만들고**, 나머지 NPU는 `_schedule_existing`으로 같은 batch(같은 workload 폴더)에 합류한다 (`scheduler.py:90-131`, `340-349`).
- batch 완료 조건: `start_npu`와 instance의 마지막 NPU가 모두 `batch.end`에 들어와야 함 (`scheduler.py:370-375`).
- DP group은 Python 쪽 barrier(`dp_pending`)에서 모든 멤버가 batch를 낼 때까지 trace 생성을 미루고, 가장 큰 `total_len`으로 padding 후 공유 workload 폴더에 그래프를 쓴다 (`__main__.py:733-931`).

### 3.3 종료 (`__main__.py:1107-1263`)

모든 인스턴스가 비면 `exit`을 보내고, `controller.check_end()`로 ASTRA-Sim의 종료 메시지를 확인한 뒤 throughput / prefix hit / power / 인스턴스별 TTFT·TPOT·ITL을 출력하고 `--output` CSV를 쓴다. `--keep-inputs`가 없으면 `astra-sim/inputs/runs/<run_id>/`를 삭제한다.

### 3.4 한 batch가 시간으로 바뀌는 경로 (요약)

```
Scheduler._build_batch()                          scheduler.py:272
  └ Batch(total_len, q_list, k_list, prefill_q_list, prefill_k_list, decode_k_list,
          load=recall_bytes, evict=0, pd_kv_send_tokens)
generate_trace()                                  trace_generator.py:1606
  ├ _synthesize_trace()                           trace_generator.py:1397
  │   ├ _emit_prologue (embedding: input_loc=REMOTE)
  │   ├ for block: _build_transformer_block → _emit_sequence → _emit_layer
  │   │             (latency = perf CSV 보간, sizes = calculate_sizes, weight_loc = placement)
  │   └ _emit_final_layers (sampler: output_loc=REMOTE)
  └ kv_load / kv_evict row prepend (batch.load / batch.evict != 0 일 때)
generate_graph() → LLMConverter.convert_rows()    graph_generator.py:138, chakra llm_converter.py
  └ llm.<npu>.et (COMP / MEM_LOAD / MEM_STORE / COMM_COLL / SEND/RECV 노드)
ASTRA-Sim Workload 실행 → "sys[X] iteration N finished, C cycles"
```

---

## 4. Python 프론트엔드 모듈별 상세 (`serving/`)

### 4.1 `config_builder.py` — 클러스터 설정 → ASTRA-Sim 입력

`build_cluster_config(astra_sim, cluster_config_path, enable_local_offloading, enable_attn_offloading, inputs_root)` (`config_builder.py:320`)

**생성하는 파일 3종** (`astra-sim/inputs/runs/<run_id>/` 아래):

| 파일 | 생성 위치 | 내용 |
| --- | --- | --- |
| `network/network.yml` | `_create_network_config` (`:719`) | `topology: FullyConnected × dims`, `npus_count`, `bandwidth`(=`link_bw`), `latency`(=`link_latency`). dims는 `_compute_network_dims` (`:230`)가 TP/PP/DP 구성으로 계산 |
| `system/system.json` | 템플릿 복사(`:28-52`) 후 수정 | `local-mem-bw` ← `npu_mem.mem_bw` (`:565`), collective 구현 배열 길이를 dim 수에 맞춤 (`_sync_system_collective_dims`, `:306`) |
| `memory/memory_expansion.json` | `:336-366`, `:534-544`, `:571-576`, `:684-686` | 아래 표 |

`memory_expansion.json`의 키 (Python이 쓰는 그대로):

| 키 | 생성 조건 | 내용 |
| --- | --- | --- |
| `remote_mem` | 항상 (첫 번째 노드의 `cpu_mem` 기준) | `memory-type: "PER_NODE_MEMORY_EXPANSION"`, `mem-bw`, `mem-latency`, `num-devices: num_nodes`, (PIM 시) `pim-channels` |
| `cxl_mem` | 최상위 `cxl_mem` 블록이 있을 때 | `memory-type: "MEMORY_POOL"`, `mem-bw`, `mem-latency`, `num-devices` |
| `local_mem` | `--enable-local-offloading`일 때 | `memory-type: "PER_NPU_MEMORY_EXPANSION"`, `mem-bw`/`mem-latency` ← `npu_mem` |

> "only one type of ... config is supported for now" 주석대로 `remote_mem`은 첫 노드의 `cpu_mem`, `local-mem-bw`는 (노드별 첫 인스턴스가 덮어쓰므로) 사실상 마지막 노드 첫 인스턴스의 `npu_mem.mem_bw` 하나로 전역 설정된다 (`:534-577`).

**PIM 설정 덮어쓰기** (`:483-506`): attention offloading이 켜지면 `cpu_mem.pim_config`(예: `DDR4_8GB_3200_pim`)로 `PIMModel`을 만들고, `cpu_mem.mem_bw`/`mem_latency`를 PIM 모델 값으로 덮어쓴다.

**병렬도 해석**: `_resolve_parallelism` (`:55`) — `num_npus = tp_size * pp_size`, MoE면 `ep_size` 기본값 = `tp_size`. `_resolve_dp_groups` (`:139`) — 같은 `dp_group` 문자열끼리 묶고 `tp_dim`/`ep_dim`(collective의 `involved_dim`)을 계산.

**NPU id 매핑** (`:587-606`): 인스턴스 순서대로 NPU id를 연속 할당. `pd_type == "prefill"`인 인스턴스는 KV를 받을 decode 측 NPU까지 포함해 `num_npus * 2`개를 차지한다.

**Placement (weight/KV 위치 지정)** (`:608-664`): 인스턴스별로 `default` → `blocks`(블록 범위 규칙) → `layers`(레이어 이름 규칙) 우선순위로 해석.

```python
# config_builder.py:863
def _mem_str(loc, node_id):
    if loc.upper().startswith("NPU"):  return "LOCAL"
    elif loc.upper().startswith("CPU"): return f"REMOTE:{node_id}"
    elif loc.upper().startswith("CXL"): return loc.upper()      # "CXL:0" ...
    else: raise ValueError(...)
```

- 결과 구조: `{"default": {...}, "block": [num_hidden_layers개 dict], "layer": {name: {...}}}` + `block_mode_on`(블록별 규칙이 있으면 True → trace를 블록마다 따로 생성, 복사 최적화 불가).
- `get_device(placement, block_idx, layer_name, kind)` (`:801`): `kind ∈ {weights, kv_loc, kv_evict_loc}`. layer 규칙 > block 규칙 > default.
- `_validate_memory_config` (`:742`): `memory_expansion.json`의 키 prefix(`remote_mem`→`REMOTE`, `cxl_mem`→`CXL`, `local_mem`→`LOCAL`)와 `num-devices`로 허용 위치 집합(`REMOTE:0`, `CXL:3` …)을 만들고 placement 값이 그 안에 있는지 검사. `LOCAL`은 local offloading이 꺼져 있으면 항상 허용.

### 4.2 `router.py` — 요청 로딩/라우팅

- `load_requests` (`:103`): JSONL을 읽어 `_pending_requests`에 넣음. flat 요청(`input_toks`, `output_toks`, `arrival_time_ns`, 선택적으로 `input_tok_ids`/`output_tok_ids`)과 agentic session(`sub_requests[]`, `tool_duration_ns`)을 모두 지원. **주의**: 내부에서 `output_toks`는 `input + output`(총 길이)으로 저장된다 (`:146`).
- `route_arrived_requests(current)` (`:189`): 도착 시각이 지난 요청을 정책(`LOAD`/`RR`/`RAND`/`CUSTOM`)에 따라 prefill-capable 스케줄러에 `add_request`.
- `notify_request_completed` (`:237`): agentic session의 다음 sub-request를 `완료시각 + tool_duration_ns`에 release.
- `transfer_prefill_request` (`:324`): P/D 분리 시 prefill이 끝난 요청을 decode 인스턴스의 `add_decode`로 넘김.

### 4.3 `scheduler.py` + `request.py` — vLLM V1 스케줄러

`Request` (`request.py:19`) 핵심 필드:

| 필드 | 의미 |
| --- | --- |
| `input` / `original_input` | 프롬프트 길이 |
| `output` | **총 목표 길이** (prompt + generated) |
| `num_computed_tokens` | KV가 계산된 토큰 수 (schedule 시점에 증가, preemption 시 0으로 리셋) |
| `num_tokens_reached` | 지금까지 도달한 길이(prompt + 생성). `num_tokens` property |
| `is_init` | 첫 토큰(TTFT) 전인지 |
| `input_hash_ids`, `block_hashes` | prefix caching용 토큰 id와 chained block hash |
| `npu_cache_hit`, `storage_cache_hit` | tier별 prefix hit 토큰 수 |

`Batch` (`request.py:118`) 핵심 필드: `total_len`(이번 step에 계산할 토큰 합), `q_list`/`k_list`(요청별 새 토큰/기존 KV 길이), `prefill_q_list`/`prefill_k_list`/`decode_k_list`, `load`(하위 tier에서 recall할 bytes), `evict`(항상 0), `write_through`, `pd_kv_send_tokens`, `fired`/`end`(이 batch를 실행 시작/완료한 NPU 목록), `workload_name`.

`Scheduler.schedule(current, sys, batch_id)` (`scheduler.py:90`):

1. start NPU가 아니면 `_schedule_existing` (기존 batch 합류)만.
2. `len(inflight) >= pp_size`면 None (PP 깊이만큼만 동시 batch).
3. **Phase A** `_schedule_running` (`:133`): running 요청을 먼저 처리. `kv.allocate_slots` 실패 시 running 꼬리부터 preempt.
4. **Phase B** `_schedule_waiting` (`:179`): 이번 step에 preempt가 없을 때만. `kv.get_computed_blocks`로 prefix hit 조회 → `can_fit_full_sequence`(reserve_full_isl) → `allocate_slots`. 대기 요청 때문에 preempt하지 않음.
5. `_build_batch` (`:272`): 토큰 수 > 1이면 prefill chunk, == 1이면 decode로 분류. `num_computed_tokens`를 schedule 시점에 증가. `kv.take_traffic()`의 recall bytes를 `batch.load`로.

`add_done(id, sys, finish)` (`:353`): batch 완료 시 TTFT/ITL/latency 기록, 토큰 생성(`num_tokens_reached += 1`), prefix cache block 인덱싱, 끝난 요청의 KV 해제.

> prefill/decode "phase"가 따로 없다. 요청은 `num_tokens_reached`까지 따라잡을 뿐이고, trace 생성은 **이번 step의 scheduled token 수**로 prefill(>1)/decode(==1)를 구분한다.

### 4.4 `memory_model.py` / `block_pool.py` / `kv_cache_manager.py` — 메모리 계층

#### Device enum (`block_pool.py:39-42`)

```python
class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3
```

(Python 쪽 KV pool 단위의 tier 구분. ASTRA-Sim trace의 위치 문자열 `LOCAL/REMOTE/CXL`과는 별개의 체계다 — 매핑은 [§8](#8-메모리-계층-모델-정리-hbf-개조-관점).)

#### `MemoryModel` (`memory_model.py:19`)

| 항목 | 계산 | 위치 |
| --- | --- | --- |
| per-GPU weight | `embedding + per_block × (n_layer // pp) + final_layernorm + lm_head` (dense는 TP로, MoE expert는 EP로 나눔) | `get_weight` `:157`, `_get_weight_per_block` `:187` |
| weight > NPU 메모리면 에러 | | `:64-66` |
| KV bytes/token (rank당) | `2 * kv_dim * n_layer * kv_fp // num_npus` | `get_kv` `:209` |
| NPU KV block 수 | `(npu_mem * mem_util - weight) // (bytes_per_token * block_size)` | `:81-94` |
| 하위 tier pool | prefix caching + `--prefix-storage CPU/CXL`일 때만. 블록 = 256 토큰 chunk(LMCache 기본), full-cluster bytes | `:113-117`, `build_prefix_pool` `:383` |
| `npu_used` / `cpu_used` | property. 예약(weight 등) + pool used bytes | `:253-269` |
| `allocate(size, device)` | KV 외의 예약 (weight, PIM 버퍼 등) | `:271` |

> **중요**: weight는 placement와 상관없이 **항상 NPU 메모리에서 차감**된다. `MemoryModel`은 placement를 전혀 참조하지 않는다 (`grep placement serving/core/memory_model.py` → 없음).

`calculate_sizes(model, layer_name, length, kv_len=None, pim=False, parallel=1, fp=2)` (`memory_model.py:427`): 레이어별 rank당 **input / weight / output bytes**. trace의 `input_size`/`weight_size`/`output_size`가 여기서 나온다.

| 레이어 | input | weight | output |
| --- | --- | --- | --- |
| `embedding` | `length*fp*2` | `(vocab/p)*h*fp` | `length*h*fp` |
| `layernorm` 등 | `length*h*fp` | `h*fp` | `length*h*fp` |
| `qkv_proj` | `length*h*fp` | `h*((q_dim+2kv_dim)/p)*fp` | `length*((q_dim+2kv_dim)/p)*fp` |
| `rotary_emb` | `(n_head/p + kv_head/p)*length*head_dim*fp` | 0 | 동일 |
| `attention` (NPU) | `(n_head/p)*length*hd*fp + (kv_head/p)*kv_len*hd*fp*2` | 0 | `(n_head/p)*length*hd*fp` |
| `attention` (`pim=True`) | 1토큰 기준 Q + K + V | 0 | 1토큰 기준 |
| `o_proj` | `length*(q_dim/p)*fp` | `(q_dim/p)*h*fp` | `length*h*fp` |
| `gate_up_proj` | `length*h*fp` | `h*2*(ffn/p)*fp` | `length*2*(ffn/p)*fp` |
| `act_fn` | `length*2*(ffn/p)*fp` | 0 | `length*(ffn/p)*fp` |
| `down_proj` | `length*(ffn/p)*fp` | `(ffn/p)*h*fp` | `length*h*fp` |
| `moe` | `length*h*fp` | `h*E*fp + (E/p)*3*h*moe_ffn*fp` | `length*h*fp` |
| `lm_head` | `length*h*fp` | `h*(vocab/p)*fp` | `length*(vocab/p)*fp` |
| `sampler` | `length*(vocab/p)*fp` | 0 | `length*4` |

> attention의 KV 읽기량은 `input_size`의 두 번째 항으로만 표현되고, KV는 "weight"가 아니다 (`weight_size = 0`). 즉 KV 위치를 바꿔도 현재 trace에는 반영될 필드가 없다.

#### `BlockPool` (`block_pool.py:210`)

vLLM `block_pool.py` 포팅. tier 하나의 free list(`FreeKVCacheBlockQueue`, 이중 연결 리스트), `cached_block_hash_to_block`(prefix index), ref count를 소유한다. `get_new_blocks`에서만 eviction이 일어나며 **eviction 비용은 0** (`:305-329`). `cache_copy` (`:281`)는 하위 tier에 inclusive write-through 복사본을 둔다.

#### `TieredKVCacheManager` (`kv_cache_manager.py:74`)

- NPU block 크기로 chained hash를 한 번 만들고, 하위 tier는 `factor = pool.block_size // npu_block_size`번째 hash마다 key로 사용 (`_coarse_hash` `:117`).
- `get_computed_blocks(req)` (`:128`): NPU hit 블록 + 하위 tier에서 추가로 복구 가능한 토큰 수.
- `allocate_slots` (`:242`): all-or-nothing. 하위 tier hit이 있으면 `_charge_recall` (`:297`)로 **recall bytes 누적 → critical path에 과금**.
- `_write_down` (`:339`): NPU → 하위 tier write-through. **bytes만 기록하고 latency는 과금하지 않음** (off-critical-path 가정; 에너지용).
- `take_traffic()` (`:384`): `(recall_bytes, write_through_bytes)`를 꺼내고 리셋 → `Scheduler._build_batch`가 `batch.load`로 사용.

### 4.5 `trace_generator.py` — 레이어별 trace 생성 (핵심)

#### (1) 성능 DB 로드/조회

- 경로: `../profiler/perf/<hardware>/<model>/<variant>/tp<N>/` (`_variant_root` `:78`). `variant`는 dtype + kv dtype (`bf16`, `bf16-kvfp8` …, `resolve_variant` `:51`).
- `_load_perf_db` (`:351`): `meta.yaml`, `skew_fit.csv`, 아키텍처 yaml(`profiler/models/<model_type>.yaml`)을 읽어 프로세스 전역 캐시(`_perf_db_cache`)에 저장.
- 조회 함수 (전부 **profiled `time_us` → ns 변환 후 선형 보간/외삽**):

| 카테고리 | 함수 | 키 |
| --- | --- | --- |
| dense | `_lookup_dense` `:550` | `tokens = total_len` |
| per_sequence | `_lookup_per_sequence` `:561` | `sequences = lm_head_len` (요청 수) |
| attention | `_lookup_attention_with_skew` `:780` → `_lookup_attention` `:827` | `(prefill_chunk, kv_prefill, n_decode, kv_decode_mean)` 4D 선형 보간 + decode KV 길이 skew 보정(alpha) |
| moe | `_lookup_moe` `:864` | `(local_tokens, activated_experts)`, 항상 tp=1 프로파일 |

#### (2) 컨텍스트

- `TraceCtx` (`:87`): 하드웨어, perf_db, placement, PIM 모델, TP/PP/EP 정보, `tp_dim`/`ep_dim` 등 trace 전체 공통 정보.
- `BatchCtx` (`:117`): `total_len`, `prefill_chunk`, `kv_prefill`, `n_decode`, `kv_decode_mean/max/min`, `lm_head_len`, PIM용 `decode_lens`/`channel_split`. `_build_batch_ctx` (`:938`)에서 생성. **PIM offload가 켜지면 여기서 `n_decode` 등을 0으로 만들어 NPU attention에서 decode 부분을 제거**한다 (`:961-971`).

#### (3) 레이어 emit

`_emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag, layer_num, comm_type, comm_size, input_loc='LOCAL', output_loc='LOCAL')` (`:993`) — **trace row 하나를 만드는 유일한 일반 경로**:

```python
# trace_generator.py:1004-1029 (요약)
latency_ns = _lookup_{per_sequence|attention_with_skew|dense}(...)     # 실측 기반 latency
inp, wt, out = calculate_sizes(ctx.model, layer_name, bctx.total_len, [kv_len], parallel=ctx.tp_size, fp=ctx.fp)
wt_loc = get_device(ctx.placement, layer_num, layer_name, "weights")    # "LOCAL" | "REMOTE:n" | "CXL:k"
lines.append((layer_name, str(latency_ns), input_loc, str(inp), wt_loc,
              str(wt), output_loc, str(out), comm_type, str(comm_size), batch_tag))
```

- `input_loc`/`output_loc`는 기본 `LOCAL`. 첫 레이어(embedding) input만 `REMOTE:{node}` (`_emit_prologue` `:1374`), 마지막 레이어(sampler) output만 `REMOTE:{node}` (`_emit_final_layers` `:1337`).
- `_emit_sequence` (`:1234`): 아키텍처 yaml의 `sequence` 리스트를 순회.
  - `attention`이면: PIM offload 시 `_emit_pim_attention` (`:1078`) 먼저, 그다음 `_emit_npu_attention` (`:1098`, prefill 부분만 남아 있으면 emit).
  - `o_proj`, `down_proj` 뒤에는 TP>1이면 `ALLREDUCE` (`_TP_ALLREDUCE_AFTER` `:42`, `_tp_comm` `:1062`), `involved_dim`은 `ALLREDUCE:1,0` 형식 (`_with_dim` `:1070`).
  - P/D prefill 인스턴스의 `qkv_proj`는 `comm_size`에 레이어별 KV 전송량을 담는다 (`_pd_kv_send_bytes` `:1047`).
- `_emit_moe_block` (`:1105`): `EXPERT {i} <dispatch comm>` … `expert` row(EP rank별) … `EXPERT END <combine comm>`.

#### (4) 전체 trace 조립

`_synthesize_trace` (`:1397`):

```
prologue (embedding)                                     ← input_loc = REMOTE:{node}
for layer in transformer blocks:                          ← block_mode_on이 아니면 1개 블록만 만들어 num_layers번 복사
    pre_attn:  layernorm, qkv_proj, rotary_emb, attention
    post_attn: o_proj(+ALLREDUCE), layernorm
    mlp_dense: gate_up_proj, act_fn, down_proj(+ALLREDUCE)   또는  mlp_moe: moe 블록
head: final_layernorm, lm_head, sampler                  ← sampler output_loc = REMOTE:{node}
```

(Llama 기준 `profiler/models/llama.yaml`의 `sequence`. 모델마다 yaml이 다르다.)

`_synthesize_interleaved_trace` (`:1460`): PIM sub-batch interleaving. batch를 둘로 나눠(`_make_sub_batch` `:1912`) `BATCH_1`/`BATCH_2` 태그로 post_attn/pre_attn을 교차 배치 → NPU 연산과 PIM attention이 overlap되도록.

`generate_trace(...)` (`:1606`) — 공개 진입점:

1. 위 trace 합성.
2. `_pp_stage_boundaries` (`:1558`): PP>1이면 블록 경계에서 stage 분할 인덱스 계산.
3. **`batch.load != 0`이면 `kv_load` row, `batch.evict != 0`이면 `kv_evict` row를 맨 앞에 추가** (`:1680-1690`):
   ```python
   load = ["kv_load", '0', 'LOCAL', '0', get_device(placement, None, None, 'kv_evict_loc'), str(load_size), 'LOCAL', '0', 'NONE', '0', 'NONE']
   ```
   → comp_time 0, **weight_loc 자리에 KV가 있는 tier, weight_size 자리에 전송 bytes**를 넣어 메모리 노드로 변환되게 하는 트릭.
4. 헤더: `COLOCATED|PREFILL|DECODE\t\tmodel_parallel_NPU_group: {pp}\t\tpp_stage_boundaries: ...` (`:1705`).
5. `TraceData(header_line, rows, path)` 반환 (`:1723`). 텍스트 파일은 `--save-trace-text`일 때만 기록 (`write_trace` `:1770`).

`generate_event(alarm)` (`:1855`): 첫 요청까지 기다리는 1-row trace (`event_{alarm}ns`, comp_time = alarm).

#### (5) Trace row 포맷 (11 필드)

```
Layername  comp_time  input_loc  input_size  weight_loc  weight_size  output_loc  output_size  comm_type  comm_size  misc
embedding_0  5621     REMOTE:0   40          LOCAL       1050673152   LOCAL       81920        NONE       0          NONE
```

| 필드 | 값 |
| --- | --- |
| `comp_time` | ns (정수) |
| `*_loc` | `LOCAL`, `REMOTE:{node}`, `REMOTE:{node}.{pim_ch}`(PIM), `CXL:{id}` |
| `comm_type` | `NONE`, `ALLREDUCE`, `ALLGATHER`, `REDUCESCATTER`, `ALLTOALL` (+ `:1,0` 형태 involved_dim) |
| `misc` | `NONE` / `BATCH_1` / `BATCH_2` |
| 마커 row (1필드) | `EXPERT {i} {comm} {size}`, `EXPERT END {comm} {size}`, `PIM {ch}`, `PIM END` |

텍스트 컬럼 폭은 `utils._FMT` (`utils.py:19`). 레이어 이름 뒤 `_{row index}`는 `indexed_cols` (`:1741`)/`_write_trace`가 붙인다.

### 4.6 `pim_model.py` — PIM attention offload (HBF 개조의 가장 가까운 선례)

- `PIMModel(node_id, mem_size, pim_config_path)` (`:48`): `configs/pim/<name>.ini`에서 채널 용량(`ch_capacity`, GB), 채널 BW(`bus_width/8 * data_rate/1000` GB/s), 읽기 latency(`CL * tCK`)를 계산.
- `get_config()` → `{mem_size, mem_bw, mem_latency, dimm_size}`. 채널 수 = `mem_size // dimm_size`.
- `get_pim_latency(n_head, kv_head, head_dim, L, channel_split)` (`:120`): **선형 회귀 모델** `(slope * L + intercept) / channel_split` ns. 스펙 이름(`DDR4_8GB_3200_pim` 등 4종)별 Llama-3.1-8B 기준 계수를 GQA 비율/KV 크기로 스케일 (`:143-185`). 등록 안 된 스펙은 `ValueError`.
- trace 측 흐름:
  1. `_build_batch_ctx`에서 `_attn_load_balancer` (`:1883`)로 decode 요청을 PIM 채널에 greedy 분배, NPU attention의 decode 항 제거.
  2. `_emit_pim_attention` (`:1078`): 채널마다
     ```
     PIM {ch}
     attention  <pim_lat>  REMOTE:{node}.{ch}  <inp>  <wt_loc>  0  REMOTE:{node}.{ch}  <out>  NONE  0  <tag>
     ...
     PIM END
     ```
  3. 그 뒤 `_emit_npu_attention`이 prefill 부분만 NPU에서 계산.

> 즉 "연산을 NPU 밖 메모리 디바이스로 옮긴다"는 개념이 이미 **(a) Python analytical latency 모델 + (b) trace의 `PIM` 블록 마커 + (c) 위치 문자열 `REMOTE:{node}.{ch}` + (d) converter/ASTRA-Sim의 PIM 처리**로 구현되어 있다. HBF 내 연산(near-flash compute)을 모델링한다면 이 경로를 템플릿으로 쓰는 것이 자연스럽다.

### 4.7 `graph_generator.py` / `controller.py`

- `generate_graph(batch, hardware, num_npus, node_id, instance_id, npu_offset, enable_local_offloading, event, workload_name, inputs_root, save_trace_text, *, trace)` (`graph_generator.py:138`):
  - 출력: `inputs/runs/<run_id>/workload/<hw>/<model>/instance<i>_batch<b>/llm.<npu>.et`
  - Chakra `LLMConverter`를 **in-process**로 import해서 `convert_rows(header, indexed_cols(rows))` 호출 (`:181-184`). (예전에는 서브프로세스였음.)
  - **설치된 chakra(site-packages)를 import**한다. converter 코드를 고치면 `astra-sim/extern/graph_frontend/chakra`에서 `pip3 install .`을 다시 해야 반영된다 (`scripts/compile.sh`).
  - trace 내용 + `(num_npus, npu_offset, enable_local_offloading)` 해시 기반의 `.et` 캐시 (`:35-100`). converter 입력을 늘리면 cache key에도 넣어야 한다.
- `get_workload(...)` (`utils.py:35`): ASTRA-Sim에 보낼 경로 문자열 `.../workload/<name>/llm` (NPU id 접미사는 ASTRA-Sim이 붙임).
- `Controller` (`controller.py`): `read_wait`(“Waiting”까지 읽기), `parse_output`(정규식 `sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles.`), `write_flush`, `check_end`("All Request Has Been Exited" / "ERROR: Some Requests Remain").

### 4.8 기타

- `gate_function.py`: `GateRouter.route_ep` — MoE token을 EP rank에 분배 (`BALANCED`는 결정적 pigeonhole 근사). expert 소유 rank = `expert_id * ep_size // num_experts`.
- `power_model.py`: NPU active/standby/idle, CPU, DRAM(`energy_per_bit` × bytes), link, NIC, storage 에너지 누적. trace_generator의 `PowerAccumulator`가 `wt_loc != 'LOCAL'`인 weight bytes를 DRAM 에너지로 더한다 (`trace_generator.py:1031-1042`).

---

## 5. Trace → Chakra 그래프 변환

파일: `astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py` (이하 `CONV`, chakra 서브모듈 @ `30221ab`)

### 5.1 진입점과 출력

- `convert_rows(header_line, rows)` (`CONV:145-191`): `graph_generator.py:184`가 in-process로 호출. 헤더의 모드로 분기:
  - `COLOCATED` / `DECODE` → `convert_common` (`:368-706`)
  - `PREFILL` → `convert_prefill` (`:708-986`)
  - `EVENT` → `convert_event` (`:988-1000`, 의존성 없는 COMP 하나 = 타이머)
- 출력: NPU마다 `llm.<npu_id>.et` (protobuf). 스키마 `chakra/schema/protobuf/et_def.proto`. 생성 파일(`et_def_pb2*.py`, `et_def.pb.cc`)은 git에 없고 `pip3 install .` / 빌드 때 생성된다.
- 노드 타입 (`et_def.proto:108-118`, LLMServingSim이 `PIM_COMP`를 추가해 upstream과 값이 다름):
  `INVALID=0, METADATA=1, MEM_LOAD=2, MEM_STORE=3, PIM_COMP=4, COMP=5, COMM_SEND=6, COMM_RECV=7, COMM_COLL=8`

### 5.2 위치 문자열 인코딩

```python
# CONV:14-19
class MemoryType(Enum):
    INVALID_MEMORY = 0
    LOCAL_MEMORY = 1
    REMOTE_MEMORY = 2
    CXL_MEMORY = 3
    STORAGE_MEMORY = 4
```

- `get_mem_type` (`:262-272`): `:` 앞 prefix로 매핑. 모르는 문자열은 `0`(INVALID) → C++에서 "Invalid memory type"으로 종료.
- `get_mem_device` (`:274-289`): `"REMOTE:1.3"` → 1, `get_mem_channel` (`:292-309`): `"REMOTE:1.3"` → 3 (PIM 채널).
- 이 enum 값은 C++ `MemoryLocationType` (`astra-sim/astra-sim/system/AstraMemoryAPI.hh:13-19`)과 **반드시 일치**해야 한다.

### 5.3 노드 빌더 — 어떤 정보가 C++로 넘어가는가

| 빌더 | 위치 | 붙는 정보 |
| --- | --- | --- |
| `get_comp_node` | `:210-213` | `duration_micros = comp_time` (**실제 단위는 ns**). **weight/input/size 속성은 없다** |
| `get_memory_load_node` / `get_memory_store_node` | `:312-324` | `tensor_size`, `tensor_loc`(MemoryType), `tensor_device` |
| `get_pim_compute_node` | `:326-333` | 위 3개 + `tensor_channel`, `duration_micros = comp_time` |
| `get_comm_coll_node` | `:226-234` | `comm_type`, `comm_size`, (선택) `involved_dim` BoolList. 모르는 comm_type은 **조용히 ALL_REDUCE(0)** |
| `get_comm_node` | `:236-260` | SEND/RECV: `comm_src`, `comm_dst`, `comm_size`, `comm_tag` (크기가 안 맞으면 에러 없이 hang) |

### 5.4 `convert_common` — COLOCATED/DECODE 변환 로직

1. **`kv_load` / `kv_evict`** (`:372-396`): 맨 앞 두 row만 이름 substring으로 검사. 각각 row의 **`weight_loc`/`weight_size`**로 MEM_LOAD / MEM_STORE를 만들고 모든 NPU 파일에 같은 노드를 기록. 각 rank의 첫 COMP/PIM이 이들에 의존.
2. **rank 배치** (`:398-415`): `npus_per_group = num_npus // num_npu_group(=pp)`, `use_comm = npus_per_group > 1` (TP collective는 TP>1일 때만).
3. **stage 입력** (`:422-453`): stage 0은 첫 row의 `input_loc`/`input_size`로 MEM_LOAD (즉 embedding 입력을 `REMOTE`에서 로드). 이후 stage는 이전 stage에서 COMM_RECV.
4. **메인 루프** (`:463-630`), 일반 row마다:
   - **weight 로드** (`:465-476`):
     ```python
     if (self.local_offloading or layers[i].weight_memory_loc != "LOCAL") and layers[i].weight_memory_size > 0:
         weight_load_node = self.get_memory_load_node(name, "WEIGHT", weight_memory_loc, weight_memory_size)
     ```
     → **weight_loc가 LOCAL이 아니거나 `--enable-local-offloading`일 때만 MEM_LOAD가 생긴다.** 이 로드 노드는 (EXPERT 블록 안을 제외하면) **부모가 없어서 t=0에 바로 issue**되고, 해당 COMP만 이 로드에 의존한다 → 사실상 "weight prefetch 스트림"으로 동작.
   - **COMP** (`:478-530`): `comp_time != 0`이고 PIM 블록 안이 아니면 생성. 이전 row의 comm 노드(없으면 comp 노드)에 체인 의존.
   - **COMM_COLL** (`:547-554`): `comm_type != "NONE" and use_comm`.
5. **EXPERT 블록** (`:558-589`): dispatch/combine COLL + `int(expert_num) % npus_per_group == npu_offset`인 rank만 expert row 실행.
6. **PIM 블록** (`:591-630`, `:533-544`, `:484-528`): `PIM {ch}` 블록을 `ch % npus_per_group == npu_offset`인 rank가 소유. 블록 안 row마다 `PIM_COMP_NODE` (`duration = comp_time`, 위치/디바이스/채널 = `input_loc`, `tensor_size = input_size + output_size`). `PIM END` 이후 첫 COMP는 PIM 노드들에 의존. `BATCH_1/BATCH_2`가 바뀌는 지점에서는 직전 PIM 블록이 아니라 그 이전 블록에 의존시켜 sub-batch 간 overlap을 만든다 (`:522-528`).
7. **stage 끝** (`:641-706`): 마지막 stage는 마지막 row의 `output_loc`/`output_size`로 MEM_STORE (sampler 결과 token id를 `REMOTE`로), 그 외 stage는 다음 stage로 COMM_SEND.
8. **PP 분할** `get_stage_edges` (`:338-366`): 헤더의 `pp_stage_boundaries` 사용.

### 5.5 `convert_prefill` — P/D 분리의 prefill 측

rank마다 파일 2개(`npu_id`, `npu_id + num_npus`)를 쓴다. 두 번째는 짝이 되는 decode NPU 역할의 proxy receiver. 이름에 `"v_proj"`가 들어간 COMP(= `qkv_proj`) 뒤에 `comm_size > 0`이면 레이어별 KV를 SEND (`:855-877`). PIM 처리는 없다.

---

## 6. ASTRA-Sim 백엔드 (C++)

### 6.1 IPC 루프 (C++ 측)

- 실행 바이너리: `astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra` (`AstraSim_Analytical_Congestion_Unaware`의 symlink).
- 메인 루프: `astra-sim/astra-sim/network_frontend/analytical/congestion_unaware/main.cc:294-478`
  1. 이벤트 큐의 한 시각 bucket 처리 (`event_queue->proceed()`).
  2. `end_npu_ids` → `start_npu_ids` 순으로 끝난(`is_finished`) NPU를 질의: stdout에 `sys[<id>] iteration <n> finished, <tick> cycles, exposed communication <x> cycles.` + `... Waiting` 출력.
  3. stdin에서 한 줄 읽음:
     - workload prefix → C++가 `.<id>.et`를 붙여 로드. start NPU는 자신과 다음 start 전까지의 NPU(관리 대상)에 로드.
     - `pass` / `pass <tick>` / `pass -1` → 재질의 억제 (`-1`은 상태 변경 → 다른 NPU도 재오픈)
     - `done` → 해당 NPU sleep, `exit` → 루프 종료
  4. 종료 시 `All Request Has Been Exited` / `ERROR: Some Requests Remain`.
- "exposed communication"은 `전역 tick - 누적 COMP tick`으로 진짜 exposed comm은 아니며 Python은 로그만 남긴다.
- ns-3 frontend(`network_frontend/ns3/AstraSimNetwork.cc`)는 이 프로토콜을 다 지원하지 않는다(정확히 `"pass"`만 처리 등) — 사실상 analytical만 사용 가능.

### 6.2 Workload 실행기 (`astra-sim/astra-sim/workload/Workload.cc`)

| 함수 | 위치 | 동작 |
| --- | --- | --- |
| 생성자 | `:30-57` | `<prefix>.<sys_id>.et` 로드, `HardwareResource(1)` |
| `issue()` | `:123-173` | `MEM_LOAD`/`MEM_STORE`/`PIM_COMP` → `issue_mem`, `COMP` → `issue_comp`, COLL/SEND/RECV → `issue_comm` |
| `issue_comp` → `issue_replay` | `:245-270`, `:175-191` | **trace의 `comp_time`(ns)을 그대로 지연으로 등록**. roofline 분기(`:248-264`)는 있지만 `roofline-enabled` 미설정 + COMP에 `num_ops`/`tensor_size` 없음 → **사용 안 됨** |
| `issue_mem` | `:207-243` | `tensor_loc`로 `sys->local_mem / remote_mem / cxl_mem / storage_mem->issue(tensor_size, wlhd)`. **load/store 구분 없이 동일 타이밍** |
| `issue_comm` | `:272-390` | `involved_dim` 없으면 `[true]*4`. `generate_all_reduce/all_to_all/all_gather/reduce_scatter`, SEND/RECV는 `front_end_sim_send/recv` |
| `call()` | `:397-478` | 완료된 노드의 자식 issue. 끝나면 `pending_workloads`의 다음 workload 로드 또는 `is_finished = true` |
| `report()` | `:564-569` | Python이 파싱하는 iteration 라인 출력 |

```cpp
// Workload.cc:175-191 (요약) — COMP 시간은 trace 값 그대로
uint64_t runtime = 1ul;
if (node->runtime() != 0ul) runtime = node->runtime();   // ns
sys->register_event(this, EventType::General, wlhd, runtime);
```

`HardwareResource.cc`: NPU당 **COMP 슬롯 1개 + comm 슬롯 1개**(COLL, SEND 공유). `MEM_LOAD`/`MEM_STORE`/`PIM_COMP`/`COMM_RECV`는 슬롯을 차지하지 않는다 ("concurrent memory access is handled in Memory Backend"). ET feeder는 node id가 작은 것부터 issue (min-heap, `chakra/src/feeder/et_feeder.h`).

→ 결과적으로 **부모 없는 weight MEM_LOAD들은 모두 t=0에 한꺼번에 issue**되고, 메모리 백엔드의 큐(아래)가 대역폭을 직렬화한다.

### 6.3 메모리 위치 enum과 `Sys`

```cpp
// astra-sim/astra-sim/system/AstraMemoryAPI.hh:13-19
enum class MemoryLocationType : uint8_t {
  INVALID_MEMORY = 0, LOCAL_MEMORY = 1, REMOTE_MEMORY = 2, CXL_MEMORY = 3, STORAGE_MEMORY = 4 };
```

- `astra-sim/astra-sim/common/AstraMemoryAPI.hh`는 **같은 include guard를 가진 낡은 사본**(enum 없음). 수정은 `system/` 쪽에.
- `Sys.hh:259-263`: `double local_mem_bw; AstraMemoryAPI *local_mem, *remote_mem, *cxl_mem, *storage_mem;` — **nullptr 초기화가 없다.** 설정 안 된 tier로 issue하면 정의되지 않은 동작(segfault 가능).
- `Sys.cc:164-189`: 메모리 객체를 `get_memory_location_type()`로 각 포인터에 연결하고 `set_sys(id, this)`. **STORAGE 분기만 `set_sys`를 호출하지 않는다.**
- `WorkloadLayerHandlerData.hh:18-24`: `sys_id, workload, node_id, device_id, pim_enabled, pim_channel_id, pim_runtime`.
- `system.json`의 `local-mem-bw`(GB/s → ×1e9, `Sys.cc:497-500`)는 **(a) 꺼져 있는 roofline과 (b) collective의 local reduction 지연(`PacketBundle.cc:52-64`)에서만 쓰인다.** COMP 시간이나 메모리 노드 시간에는 영향이 없다.

### 6.4 메모리 백엔드 (`astra-sim/extern/memory_backend/analytical/`)

**설정 파싱** (`congestion_unaware/main.cc:107-157`, 같은 코드가 `congestion_aware/main.cc`, `ns3/AstraSimNetwork.cc`에도 복제됨):

- 최상위에 `memory-type`/`mem-latency`/`mem-bw`가 있으면 단일 tier(REMOTE).
- 아니면 **`local_mem`, `remote_mem`, `cxl_mem` 키만** 인식. 각각에 `"memory-location": "LOCAL_MEMORY" | "REMOTE_MEMORY" | "CXL_MEMORY"`를 주입해 `AnalyticalMemory`를 하나씩 생성. (`storage_mem` 키는 없음 → STORAGE는 설정 불가)
- **tier당 `AnalyticalMemory` 객체 1개를 클러스터의 모든 `Sys`(NPU)가 공유**한다.

`AnalyticalMemory` (`AnalyticalMemory.hh:49-75`, `.cc`):

- 읽는 JSON 키: `memory-type`, `memory-location`, `mem-latency`(ns), `mem-bw`(GB/s), `num-devices`, `pim-channels`. `mem_bw`/`mem_latency`는 **`uint64_t`라 소수점이 잘린다.**
- `enum MemoryArchitectureType { NO_MEMORY_EXPANSION, PER_NODE_MEMORY_EXPANSION, PER_NPU_MEMORY_EXPANSION, MEMORY_POOL }`

**타이밍 공식** (`AnalyticalMemory.cc:271-275`):

```cpp
uint64_t AnalyticalMemory::get_mem_runtime(uint64_t tensor_size) {
  return mem_latency + (uint64_t)((double)tensor_size / mem_bw);   // ns = bytes / (GB/s), 1GB = 1e9B
}
```

**큐/경합 모델** (`AnalyticalMemory::issue`, `:126-200`):

| memory-type | 사용처 | 경합 모델 |
| --- | --- | --- |
| `PER_NODE_MEMORY_EXPANSION` | `remote_mem` (CPU) | `device_id`(= `REMOTE:n`의 n)별 **FIFO 1개**. 바쁘면 대기열, 아니면 `now + runtime`에 완료. 요청마다 latency를 직렬로 지불 |
| `MEMORY_POOL` | `cxl_mem` | 코드상 PER_NODE와 동일 (device별 FIFO, 클러스터 전체 공유) |
| `PER_NPU_MEMORY_EXPANSION` | `local_mem` (local offloading) | **큐 없음.** 모든 요청이 독립적으로 `now + latency + size/bw`에 완료 → 동시 요청 수만큼 대역폭이 무한히 늘어나는 셈 |
| PIM (`pim_enabled`) | `remote_mem` + `pim-channels` | `(device, channel)`별 FIFO, `runtime = pim_runtime + latency + size/bw` (`:133-157`). 일반 큐와 독립 |

### 6.5 네트워크 백엔드 (요약)

`network.yml` (`extern/network_backend/analytical/common/network-parser/NetworkParser.cpp:66-84`): 차원별 `topology`, `npus_count`, `bandwidth`(GB/s, 여기서는 GiB 기준), `latency`(ns). P2P 지연 = `hops·latency + size / (BW·2^30/1e9)`, FullyConnected는 1 hop, congestion 없음. Collective는 `Sys::generate_collective`에서 차원별 ring 알고리즘으로 분해 (Python이 `["ring"] * num_dims` 강제). ring all-reduce는 `2(N-1)` 스텝 × `S/N` bytes, reduce 스텝마다 `3·(S/N)/local_mem_bw` 추가.

---

## 7. 입력/설정 파일 포맷

### 7.1 Cluster config (`configs/cluster/*.json`)

```jsonc
{
  "num_nodes": 1,
  "link_bw": 16,              // GB/s (스칼라 또는 차원별 리스트)
  "link_latency": 20000,      // ns
  "cxl_mem": {"mem_size": 1024, "mem_bw": 60, "mem_latency": 250, "num_devices": 4},   // 선택
  "nodes": [{
    "num_instances": 1,
    "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0, "pim_config": "DDR4_8GB_3200_pim"},  // pim_config는 PIM 시
    "power": { ... },         // 선택 (모든 노드에 있어야 전력 모델 활성)
    "instances": [{
      "model_name": "meta-llama/Llama-3.1-8B",     // configs/model/<이름>.json
      "hardware": "RTXPRO6000",                    // profiler/perf/<hardware>/
      "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0, "mem_util": 0.9},
      "num_npus": 1, "tp_size": 1, "pp_size": 1, "ep_size": 1, "dp_group": null,
      "pd_type": null,                             // null | "prefill" | "decode"
      "placement": {                               // 선택
        "default": {"weights": "npu", "kv_loc": "npu", "kv_evict_loc": "cpu"},
        "blocks":  [{"blocks": "0-3", "weights": "cxl:0"}],
        "layers":  {"lm_head": {"weights": "cxl:3"}}
      },
      // 인스턴스별 override 가능: max_num_seqs, max_num_batched_tokens, block_size, dtype, kv_cache_dtype,
      // enable_prefix_caching, enable_chunked_prefill, enable_attn_offloading, enable_local_offloading, ...
    }]
  }]
}
```

단위: `mem_size` GB, `mem_bw` GB/s, `mem_latency` ns.

### 7.2 생성되는 `memory_expansion.json` 예 (`single_node_cxl_instance.json` 기준)

```json
{
  "cxl_mem":    {"memory-type": "MEMORY_POOL", "mem-bw": 60, "mem-latency": 250, "num-devices": 4},
  "remote_mem": {"memory-type": "PER_NODE_MEMORY_EXPANSION", "mem-bw": 256, "mem-latency": 0, "num-devices": 1}
}
```

### 7.3 기타

| 파일 | 포맷 | 사용처 |
| --- | --- | --- |
| `configs/model/<org>/<name>.json` | HF `config.json` 부분집합 (`hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `num_hidden_layers`, `vocab_size`, `model_type`, `max_position_embeddings`, MoE 키) | `utils.get_config` (lru_cache) |
| `profiler/models/<model_type>.yaml` | `sequence:` (prologue / pre_attn / post_attn / mlp_dense / mlp_moe / head) + `catalog:` (레이어 → 카테고리) | trace 생성 순서 |
| `profiler/perf/.../tp<N>/dense.csv` | `layer,tokens,time_us` | `_lookup_dense` |
| `.../per_sequence.csv` | `layer,sequences,time_us` | `_lookup_per_sequence` |
| `.../attention.csv` | `prefill_chunk,kv_prefill,n_decode,kv_decode,time_us` | `_lookup_attention` |
| `.../moe.csv` | `tokens,activated_experts,time_us` | `_lookup_moe` |
| `.../skew_fit.csv`, `meta.yaml` | decode KV 길이 skew 보정 alpha, 프로파일 메타 | `_skew_alpha` |
| `configs/pim/*.ini` | DRAM 파라미터 (`bankgroups`, `banks_per_group`, `bus_width`, `device_width`, `columns`, `rows`, `channel_size`, `data_rate`, `CL`, `tCK`, `idle_power`, `peak_power`) | `PIMModel` |
| `workloads/*.jsonl` | flat: `{"input_toks","output_toks","arrival_time_ns","input_tok_ids","output_tok_ids"}` / agentic: `{"session_id","arrival_time_ns","sub_requests":[{...,"tool_duration_ns"}]}` | `Router.load_requests` |

번들된 프로파일: `RTXPRO6000`(Llama-3.1-8B, Qwen 계열), `RTX4090`(Llama). 새 하드웨어는 GPU에서 `profiler/profile.sh`로 만들어야 한다.

---

## 8. 메모리 계층 모델 정리 (HBF 개조 관점)

### 8.1 tier 이름 대응표

| 물리 개념 | placement 문자열 | trace 위치 문자열 | Chakra / C++ enum | `memory_expansion.json` 키 / type | C++ 경합 모델 | Python KV pool (`Device`) |
| --- | --- | --- | --- | --- | --- | --- |
| NPU HBM (near tier) | `npu` | `LOCAL` | 1 `LOCAL_MEMORY` | `local_mem` / `PER_NPU_MEMORY_EXPANSION` (**local offloading 때만 생성**) | 큐 없음 | `Device.NPU` (주 KV pool) |
| 호스트 DRAM | `cpu` | `REMOTE:{node}` | 2 `REMOTE_MEMORY` | `remote_mem` / `PER_NODE_MEMORY_EXPANSION` (항상) | node별 FIFO | `Device.CPU` (victim tier) |
| PIM (호스트 DRAM 안) | (`--enable-attn-offloading`) | `REMOTE:{node}.{ch}` | 2 + channel | `remote_mem.pim-channels` | (node, ch)별 FIFO | 없음 |
| CXL 메모리 | `cxl:N` | `CXL:N` | 3 `CXL_MEMORY` | `cxl_mem` / `MEMORY_POOL` | device별 FIFO (클러스터 공유) | `Device.CXL` (victim tier) |
| Storage | 없음 | `STORAGE` | 4 `STORAGE_MEMORY` | 없음 | **동작 안 함** | 없음 |

### 8.2 데이터/연산별로 어디서 시간이 계산되는가

| 데이터 / 연산 | 기본 동작 | 시간이 계산되는 곳 | 현재 바꿀 수 있는 방법 |
| --- | --- | --- | --- |
| dense 레이어의 **weight 읽기** | `LOCAL`. 별도 노드 없음 | **프로파일된 `comp_time` 안에 포함** (GPU 실측) | `placement.weights = cpu / cxl:N` → 레이어마다 MEM_LOAD 추가 (단, `comp_time`은 그대로라 HBM 읽기 시간이 **이중 계산**됨). `--enable-local-offloading` → LOCAL weight도 MEM_LOAD (PER_NPU, 경합 없음) |
| attention의 **KV 읽기** | NPU HBM | `comp_time` 안 (attention CSV 4D 보간. decode attention은 사실상 HBM 대역폭 bound: `time ≈ a + b·n_decode·kv_decode`) | **없음.** `kv_loc`은 파싱/검증만 되고 trace에 반영 안 됨. 유일한 우회는 PIM offload (decode attention 자체를 이동) |
| 새 토큰의 **KV 쓰기** | NPU HBM | `comp_time` 안 | 없음 |
| 레이어 간 activation | `LOCAL` | `comp_time` 안 | 없음 |
| KV **용량** | NPU pool = `mem_size × mem_util − weight` | Python `BlockPool` (블록 수) | `npu_mem.mem_size`, `mem_util`, `--block-size`, `--kv-cache-dtype` |
| 하위 tier KV **recall** | `--prefix-storage CPU/CXL` + prefix caching 시 | `batch.load` → `kv_load` row → MEM_LOAD (`kv_evict_loc` tier, FIFO) | `--prefix-storage`, `placement.default.kv_evict_loc` |
| 하위 tier **write-through** | 위와 동일 | **latency 0** (bytes만 에너지용으로 기록) | 없음 |
| NPU KV **eviction** | | 비용 0 | 없음 |
| 요청 입출력 | embedding 입력 `REMOTE` 로드, sampler 출력 `REMOTE` 저장 | MEM_LOAD/STORE (`remote_mem`) | |
| decode attention (PIM) | `REMOTE:{node}.{ch}` | `PIM_COMP`: `pim_runtime`(Python 선형 모델) `+ latency + (in+out)/bw` | `--enable-attn-offloading`, `cpu_mem.pim_config` |
| collective | | network backend (+ reduce 스텝에 `local-mem-bw`) | `link_bw`, `link_latency` |

### 8.3 핵심 해석

1. **near tier(HBM)에서 일어나는 모든 일은 "블랙박스 `comp_time`"이다.** HBM 대역폭/용량 파라미터 중 시간에 영향을 주는 것은 사실상 없다 (`npu_mem.mem_bw`는 `local-mem-bw`로 가서 collective reduction에만 쓰이고, local offloading이 켜졌을 때만 MEM_LOAD 타이밍에 쓰임). `npu_mem.mem_size`는 KV block 수(=동시 처리 가능한 요청 수, preemption)에만 영향.
2. **NPU 밖 tier는 "추가 메모리 전송 노드"로만 표현된다.** weight를 CXL에 두면 `comp_time`(HBM에서 읽는 시간 포함)은 그대로 두고 CXL→NPU 전송 시간을 별도로 붙인다. 로드는 t=0에 모두 issue되어 FIFO로 직렬 처리되므로, 한 iteration 시간은 대략 `max(COMP 체인 누적, weight 스트림 누적 전송 시간)` 형태가 된다 (레이어 i의 COMP는 자기 weight 로드가 끝나야 시작).
3. **"연산 자체를 다른 메모리 디바이스로 옮기는" 선례는 PIM뿐**이고, 그 경로는 Python analytical latency + trace 마커 + 전용 노드 타입(`PIM_COMP`) + 메모리 백엔드의 채널별 큐로 구성된다.

---

## 9. 문서와 코드가 다른 부분 / 주의사항

공식 문서(`docs/docs/...`)와 실제 코드가 다르거나, 개조 시 걸리기 쉬운 부분들. **코드가 기준**이다.

### 9.1 문서 ≠ 코드

| 문서 주장 | 실제 코드 |
| --- | --- |
| CXL 예제: "weight가 CXL에 있으므로 `npu_mem` 사용량이 크게 줄어든다" (`docs/docs/examples/memory-tiers/cxl-memory.mdx`) | `MemoryModel.get_weight()`는 placement를 보지 않는다. **weight는 항상 NPU 메모리에서 차감**되고 KV block 수도 그대로 |
| `kv_loc: "cxl:0"`으로 KV cache를 CXL에 둘 수 있다 | `kv_loc`은 `config_builder`에서 파싱/검증만 되고 **어디에서도 사용되지 않음** (`grep kv_loc serving/`) |
| PIM offload 시 "KV blocks live in PIM memory, `npu_used` drops" (`docs/docs/simulator/specialized/pim-offload.md`) | `MemoryModel`/스케줄러에 PIM 관련 분기 없음. KV는 여전히 NPU pool에서 할당 |
| PIM 채널 배정 `channel_for_head(h) = h * num_channels // num_attention_heads` | 실제는 `_attn_load_balancer` (`trace_generator.py:1883`)의 **요청 단위 greedy min-load** 분배 |
| PIM trace row 이름 `pim_attention_<i>` | 실제 이름은 `attention` |

### 9.2 동작상 주의점 (개조 시 함정)

1. **이중 계산**: profiled `comp_time`에는 HBM 접근 시간이 이미 포함되어 있다. 다른 tier에서 weight/KV를 읽는 노드를 추가하면, HBM 몫을 빼지 않는 한 메모리 시간이 두 번 들어간다.
2. **weight 로드 = t=0 prefetch**: converter의 weight MEM_LOAD는 부모가 없다 → staging 버퍼 한계(예: HBM에 몇 레이어까지만 올릴 수 있는가)를 모델링하려면 `weight_load[i] ← comp[i-k]` 같은 의존성을 converter에 추가해야 한다.
3. **`PER_NPU_MEMORY_EXPANSION`은 경합이 없다**: 동시 요청이 모두 독립 완료 → 대역폭 제한이 사실상 없음. HBF tier를 이 타입으로 만들면 대역폭 병목이 안 보인다.
4. **`kv_load`의 출발 tier는 `kv_evict_loc`** (`trace_generator.py:1682`): 기본값 `"cpu"`이므로 `--prefix-storage CXL`이어도 placement에서 `kv_evict_loc: "cxl:0"`을 주지 않으면 recall 시간은 CPU(`remote_mem`) 파라미터로 계산된다.
5. **load/store 비대칭 없음**: C++ `issue_mem`은 노드 타입을 메모리 백엔드에 넘기지 않는다. 플래시의 느린 쓰기를 모델링하려면 `WorkloadLayerHandlerData`에 플래그 추가가 필요.
6. **단위**: `mem-bw`/`mem-latency`가 `uint64_t`라 소수 잘림, `mem-bw = 0`이면 0으로 나눔. 메모리는 1GB = 1e9B, 네트워크는 GiB(2^30).
7. **STORAGE tier는 동작하지 않음**: JSON 키 없음 + `Sys`가 `set_sys` 안 함 → null `Sys*` 역참조.
8. **`Sys`의 메모리 포인터 미초기화**: 설정 안 된 tier로 issue하면 UB.
9. **chakra는 site-packages 설치본을 import**: `llm_converter.py` 수정 후 `cd astra-sim/extern/graph_frontend/chakra && pip3 install .` 필요. proto를 바꾸면 생성된 `et_def.pb.{h,cc}`를 지우고 재빌드해야 한다(없을 때만 재생성).
10. **graph 캐시 키**: `graph_generator.py:162`의 cache key는 `(trace digest, num_npus, npu_offset, enable_local_offloading)`. converter에 새 입력(예: HBF 옵션)을 추가하면 키에도 넣어야 잘못된 `.et`를 재사용하지 않는다.
11. **block copy**: placement에 `blocks` 규칙이 있으면 `block_mode_on = True` → 레이어마다 trace를 새로 생성(느려짐). 레이어별로 위치가 다르면 피할 수 없다.
12. **회귀 검사**: `./serving/validate.sh`가 모든 시나리오의 `Total clocks (ns)`를 baseline과 정확히 비교한다. 새 기능은 기본값(off)에서 기존 clock이 변하지 않게 만드는 것이 이 저장소의 관례.

### 9.3 발견된 (의심) 버그 — HBF 작업 중 마주칠 수 있음

| 위치 | 내용 |
| --- | --- |
| `llm_converter.py:596`, `:491` | `attn_remain` 판정이 `"attn" in name`인데 Python row 이름은 `"attention"`(“attn” 부분문자열 없음) → PIM 뒤 NPU prefill attention 처리 분기가 실행되지 않음 |
| `llm_converter.py:664-666` | 마지막 stage 분기에서 `output_store_node`가 아니라 `send_output_node`에 parent를 붙임 (PIM 노드가 남아 있을 때만 영향) |
| `AnalyticalMemory.cc:140` | PIM 큐 인덱스 `num_devices * device_id + ch` — `pim_channels * device_id + ch`여야 맞음. 노드 2개 이상에서 채널이 섞이거나 범위 초과 |
| `trace_generator.py:1883-1904` | PIM 분배/latency가 현재 KV 길이가 아니라 `req.input`(프롬프트 길이)을 사용 → decode가 진행돼도 PIM attention 시간이 늘지 않음 (모델링 단순화일 수 있음) |

---

## 10. HBF 개조 시 수정 지점 지도

"near tier가 하던 일을 HBF로 옮긴다"는 목표를 어떤 의미로 해석하느냐에 따라 손대야 할 층이 다르다. 아래는 해석별 **수정 지점 목록**이다 (설계 결정은 개조 작업에서).

### 10.0 공통: 새 메모리 위치 `HBF` 추가 (C++ 노드로 시간을 계산하려면 필요)

| 층 | 파일:라인 | 변경 |
| --- | --- | --- |
| cluster config | `config_builder.py:351-366` 참고 | 최상위 `hbf_mem` 블록(`mem_size`, `mem_bw`, `mem_latency`, `num_devices`, 필요 시 read/write BW 분리) 파싱 → `memory_config["hbf_mem"]` |
| placement | `config_builder.py:863-871` (`_mem_str`) | `"hbf"` → `"HBF:{id}"` 분기 추가 (없으면 `ValueError`) |
| 검증 | `config_builder.py:755-764` | 키 prefix로 자동 처리됨 (`hbf_mem` → `HBF:i`) |
| converter enum | `llm_converter.py:14-19`, `:262-272` | `HBF_MEMORY = 5`, `get_mem_type`에 `"HBF"` |
| C++ enum | `astra-sim/astra-sim/system/AstraMemoryAPI.hh:13-19` | `HBF_MEMORY = 5` (`common/` 사본은 건드리지 말 것) |
| `Sys` | `system/Sys.hh:260-263`, `system/Sys.cc:164-189` | `hbf_mem` 포인터(+ 다른 포인터 nullptr 초기화), CXL 분기를 복사해 `set_sys` 호출 |
| Workload | `workload/Workload.cc:220-242` | `case HBF_MEMORY: sys->hbf_mem->issue(...)` (+ null 체크) |
| 메모리 백엔드 | `AnalyticalMemory.cc:49-69` | `"HBF_MEMORY"` 위치 파싱. 필요 시 새 timing 모델 (아래) |
| frontend 3곳 | `congestion_unaware/main.cc:124-151`, `congestion_aware/main.cc:125-152`, `ns3/AstraSimNetwork.cc` | `hbf_mem` 키 → `"memory-location": "HBF_MEMORY"` 주입 |
| 빌드 | `scripts/compile.sh` | chakra 재설치 + `astra-sim/build/astra_analytical/build.sh` |

HBF 타이밍 모델에서 고려할 것 (현재 `AnalyticalMemory`에 없는 것들):
- **경합 단위**: NPU별 HBF라면 FIFO를 `wlhd->sys_id`로 키잉해야 한다 (TP 그룹의 모든 rank가 같은 위치 문자열을 쓰므로 `"HBF:n"`의 n으로는 NPU별 디바이스를 표현할 수 없음). 노드/풀 공유라면 기존 `MEMORY_POOL` 동작을 재사용 가능.
- **읽기/쓰기 비대칭**: `WorkloadLayerHandlerData`에 load/store 플래그를 추가하고 `issue_mem`에서 `node->type()`으로 설정.
- **페이지/블록 단위 접근**: `tensor_size`를 페이지 크기로 올림.
- **정밀도**: bw/latency를 `double`로.
- **use-after-free 주의**: 디바이스 해제 이벤트와 워크로드 완료 이벤트를 다른 시각으로 분리하면 `Workload::call`이 `wlhd`를 지운 뒤(`Workload.cc:451`) 디바이스 콜백이 읽을 수 있다 → 필요한 필드를 `PendingMemoryRequest`에 복사.

### 10.1 해석 A — "HBM을 HBF로 대체" (NPU의 near memory 자체가 HBF)

가장 단순한 1차 근사. **C++ 변경 없이 Python만으로 가능.**

- **용량**: `npu_mem.mem_size`를 HBF 용량으로 → KV block 수 증가 (`memory_model.py:81-94`).
- **시간**: `trace_generator._emit_layer` (`:993-1029`)에서 `latency_ns`를 보정. 이 함수는 이미 `inp`/`wt`/`out` bytes를 알고 있으므로, 예: `t' = max(t_compute, (wt + kv_bytes) / BW_hbf)` 또는 `t' = t + bytes·(1/BW_hbf − 1/BW_hbm)` 형태의 roofline 보정. attention은 `_lookup_attention_with_skew` 결과에 대해 decode KV 항(`kv_len_for_sizes`, `:1018`)만 따로 보정하는 것이 자연스럽다. MoE는 `_emit_moe_block` (`:1105-1190`)의 expert row도 같이.
- **KV 쓰기**: 현재 별도 항이 없으므로 decode step마다 `2·kv_dim·n_layer·kv_fp/tp` bytes 쓰기를 추가 비용으로 넣을지 결정 필요.

### 10.2 해석 B — "weight를 HBF에 두고 스트리밍" (HBM은 버퍼/캐시)

- placement `weights: "hbf:0"` → `_emit_layer`가 `weight_loc = "HBF:0"`을 기록 → converter가 이미 MEM_LOAD를 만들어 줌 (converter 로직 변경 불필요, enum만).
- **필수**: `comp_time`에서 HBM weight 읽기 몫을 빼는 보정 (§9.2-1).
- **필수에 가까움**: `MemoryModel.get_weight()` (`memory_model.py:157`)를 placement-aware로 → HBF에 둔 weight만큼 NPU KV 용량 증가. (`Scheduler` → `MemoryModel` 생성자에 placement 전달 필요, `__main__.py:528-543`)
- 선택: HBM staging 깊이 제한 → converter의 weight 로드에 의존성 추가 (`llm_converter.py:465-476`, `:802-813` prefill 경로도).

### 10.3 해석 C — "KV cache를 HBF에" (attention의 KV 읽기를 HBF에서)

- **현재 표현 수단이 없다.** 선택지:
  - (a) attention row의 `weight_loc`/`weight_size`에 `HBF:x`/KV bytes를 넣는 우회 (converter 변경 0, 의미상 꼼수). `_emit_npu_attention` (`:1098`) / `_emit_layer`에서 처리.
  - (b) 레이어마다 `kv_read` 같은 별도 row를 추가 → converter가 현재 맨 앞 2줄의 `kv_load/kv_evict`만 특별 처리하므로 (`llm_converter.py:372-396`) 일반화 필요.
  - (c) PIM 경로를 본뜬 전용 노드 (아래 D).
- **용량/할당**: KV의 주 tier를 HBF로 바꾸려면 `BlockPool(Device.HBF, ...)`를 `MemoryModel`에서 NPU pool 대신/추가로 생성 (`memory_model.py:96-122`), `Device` enum에 `HBF` (`block_pool.py:39-42`).
- **victim tier로만 쓸 경우 (가장 쉬움)**: `--prefix-storage` 선택지에 `HBF` 추가 (`__main__.py:308`, `:454-512`), `MemoryModel._build_storage_pool` (`memory_model.py:124-155`), `placement.kv_evict_loc: "hbf:0"` → 기존 `kv_load` 메커니즘이 그대로 HBF recall 시간을 계산. 단 write-through는 latency 0 가정 (`kv_cache_manager.py:339-355`) → 플래시 쓰기 비용을 넣으려면 여기도 수정.
- **attention `comp_time` 보정**: decode attention은 KV 대역폭 bound이므로 HBF BW 비율로 decode 몫을 스케일.

### 10.4 해석 D — "HBF 근처에서 연산" (near-flash processing, PIM 유사)

PIM offload 경로를 템플릿으로 그대로 복제하는 방식.

| 층 | PIM에서의 위치 | HBF 버전에서 할 일 |
| --- | --- | --- |
| CLI/설정 | `--enable-attn-offloading` (`__main__.py:313`), `cpu_mem.pim_config` (`config_builder.py:483-506`) | `--enable-hbf-offloading` 등 + `hbf_mem` 설정 |
| latency 모델 | `PIMModel.get_pim_latency` (`pim_model.py:120-185`) | `HBFModel` (내부 대역폭, 채널/die 병렬도, 연산 처리량) |
| batch 분할 | `_build_batch_ctx` (`trace_generator.py:961-971`), `_attn_load_balancer` (`:1883`) | 어떤 연산(decode attention? GEMV?)을 HBF로 보낼지와 NPU 쪽에서 뺄 몫 |
| trace emit | `_emit_pim_attention` (`:1078-1095`), `_emit_sequence` (`:1244-1248`) | `HBF {ch}` … `HBF END` 블록 emit |
| converter | `Layer.__init__` PIM 파싱 (`llm_converter.py:37-45`), PIM 워크 (`:533-544`, `:591-630`), `get_pim_compute_node` (`:326-333`) | 새 마커/노드 또는 `PIM_COMP`를 HBF 위치로 재사용 |
| C++ | `Workload::issue_mem` PIM 필드 (`:215-219`), `AnalyticalMemory::issue` PIM 분기 (`:133-157`) | HBF tier에서 채널별 큐 (큐 할당은 현재 PER_NODE/MEMORY_POOL에서만 됨, `:91-117`) + **queue_idx 버그 수정** |
| 새 노드 타입이 필요하면 | `et_def.proto:108-118`, `HardwareResource.cc` (`:51-55`, `:75-79`, `:105-111`), `Workload::issue` (`:129-131`), `et_feeder_node.cpp:25-58` | 각 목록에 추가 |
| sub-batch overlap | `_synthesize_interleaved_trace` (`trace_generator.py:1460`) | NPU 연산과 HBF 연산 overlap |
| 전력 | `PowerAccumulator.pim_latencies_ns`, `power_model.add_pim_active_energy_consumption` | HBF 전력 항 |

### 10.5 검증 루틴

1. 새 기능이 꺼진 상태에서 `./serving/validate.sh --clocks-only`가 그대로 통과하는지 (기존 동작 불변).
2. `--save-trace-text --keep-inputs`로 trace 텍스트와 `.et`를 남겨 `HBF:` 위치/마커가 기대대로 나오는지 확인.
3. HBF 파라미터를 HBM과 같게 넣었을 때 기존 결과에 근접하는지(이중 계산 여부 점검), BW를 낮췄을 때 TPOT가 단조 증가하는지.

---

## 11. 부록: 주요 함수 인덱스

| 기능 | 위치 |
| --- | --- |
| 진입점 / CLI | `serving/__main__.py:251` `main()`, `:260-375` |
| ASTRA-Sim 실행 | `serving/__main__.py:623-630` |
| 메인 루프 | `serving/__main__.py:662-1159` |
| DP barrier / padding | `serving/__main__.py:35` `_pad_batch_to_max`, `:733-931` |
| 클러스터 설정 해석 | `serving/core/config_builder.py:320` `build_cluster_config` |
| placement 조회 | `serving/core/config_builder.py:801` `get_device`, `:863` `_mem_str` |
| 메모리 설정 검증 | `serving/core/config_builder.py:742` `_validate_memory_config` |
| 네트워크 차원 계산 | `serving/core/config_builder.py:230` `_compute_network_dims` |
| 요청 라우팅 | `serving/core/router.py:189` `route_arrived_requests` |
| 스케줄링 | `serving/core/scheduler.py:90` `schedule`, `:133`, `:179`, `:272` `_build_batch` |
| 완료 처리 | `serving/core/scheduler.py:353` `add_done` |
| weight / KV 크기 | `serving/core/memory_model.py:157` `get_weight`, `:209` `get_kv`, `:427` `calculate_sizes` |
| KV pool | `serving/core/block_pool.py:210` `BlockPool` |
| tier KV manager | `serving/core/kv_cache_manager.py:74`, `:128` `get_computed_blocks`, `:242` `allocate_slots`, `:384` `take_traffic` |
| perf DB | `serving/core/trace_generator.py:351` `_load_perf_db`, `:550/:561/:780/:827/:864` lookups |
| trace row 생성 | `serving/core/trace_generator.py:993` `_emit_layer` |
| 레이어 순회 | `serving/core/trace_generator.py:1234` `_emit_sequence` |
| PIM attention emit | `serving/core/trace_generator.py:1078` `_emit_pim_attention`, `:1883` `_attn_load_balancer` |
| MoE 블록 | `serving/core/trace_generator.py:1105` `_emit_moe_block` |
| trace 합성 | `serving/core/trace_generator.py:1397` `_synthesize_trace`, `:1460` interleaved, `:1606` `generate_trace` |
| kv_load/kv_evict row | `serving/core/trace_generator.py:1680-1690` |
| PIM latency | `serving/core/pim_model.py:120` `get_pim_latency` |
| 그래프 변환 | `serving/core/graph_generator.py:138` `generate_graph` |
| IPC 파싱 | `serving/core/controller.py` `read_wait`, `parse_output` |
| converter | `chakra/src/converter/llm_converter.py:145` `convert_rows`, `:368` `convert_common`, `:708` `convert_prefill` |
| C++ 메인 루프 | `astra-sim/astra-sim/network_frontend/analytical/congestion_unaware/main.cc:294-478` |
| C++ 메모리 설정 파싱 | 같은 파일 `:107-157` |
| C++ 노드 실행 | `astra-sim/astra-sim/workload/Workload.cc:123` `issue`, `:175` `issue_replay`, `:207` `issue_mem` |
| C++ 메모리 타이밍 | `astra-sim/extern/memory_backend/analytical/AnalyticalMemory.cc:126` `issue`, `:271` `get_mem_runtime` |
| C++ 메모리 enum | `astra-sim/astra-sim/system/AstraMemoryAPI.hh:13-19` |
