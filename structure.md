# LLMServingSim 2.0 코드 구조 분석

> 목적: LLMServingSim 2.0이 어떻게 구성되어 있고, 한 iteration이 어떤 경로로 시뮬레이션되는지 정리한다.
> 이후 "near tier(NPU 로컬 HBM)가 담당하던 연산/데이터를 HBF(High Bandwidth Flash)로 옮기는" 개조 작업의 기준 문서로 쓰는 것이 목표다.
>
> - 분석 대상: `https://github.com/casys-kaist/LLMServingSim` `main` @ `a4053bc` (2026/08)
> - 서브모듈: `astra-sim` @ `d346994` (casys-kaist fork, Chakra / analytical network / analytical memory 백엔드 포함)
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

> **HBF 개조 관점에서 가장 중요한 사실**: 현재 시뮬레이터에서 near tier(HBM) 접근 시간은 **별도 이벤트로 존재하지 않고, 프로파일된 레이어 latency(`comp_time`) 안에 암묵적으로 녹아 있다.** weight를 NPU 밖(`cpu`, `cxl:N`)에 두거나 KV recall이 있을 때만 "메모리 노드"가 따로 생겨 ASTRA-Sim memory backend로 시간이 계산된다. 자세한 내용은 [§8](#8-메모리-계층-모델-정리-hbf-개조-관점).

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

<!-- BACKEND_SECTIONS -->
