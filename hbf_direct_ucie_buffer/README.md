# hbf_direct_ucie_buffer — GPU↔HBF 직결(UCIe) 시 latency 은닉 버퍼 실험

**질문**: HBF가 GPU에 UCIe로 직결되어 near tier가 되거나 weight 전용 tier가 될 때,
HBF의 긴 접근 latency(tR)를 가리고 유효 대역폭(grade 3 = 3,072 GB/s)을 뽑아내려면
GPU 쪽에 얼마만큼의 in-flight 버퍼가 필요한가? HBM처럼 L2 cache의 miss 처리만으로 되는가?

**방법**: 채널 단위 closed-loop 이산사건 시뮬레이션(`sim.py`) + Little's law 해석식.
HBF와 HBM 모두 채널이 독립이므로(OCP §4.3) 채널 결과 × 채널 수가 cube/stack 결과다.

```
host --cmd--> [bank별 page cache buffer(2×4 KiB) | sense unit(tR)] --data pipe(192 GB/s/ch)--> host
```

| 모델 요소 | HBF (OCP v0.7.0) | HBM3e (6-stack GPU급) | 출처 |
|---|---|---|---|
| 채널 | 16 × 192 GB/s 유효 (256 raw × 0.75 AXI) | 96 pseudo-ch × 50 GB/s (6 × 0.8 TB/s) | OCP Table 2 / [외부] |
| page / DLU | 4 KiB, 64 B 단위 요청, burst ≤ 4 KiB | 1 KiB row, 64 B 요청 | OCP §4.1 / [외부] |
| bank별 cache | 2 page (NCBB 기본) | open row 1개 | OCP §5.3.1 / [외부] |
| sense | tR ∈ {3, 20, 50} µs, 같은 bank 직렬 | tRCD+tRP ≈ 35 ns | [가정] / [외부] |
| 링크 왕복 고정 지연 | 0.3 µs + hit 0.3 µs | fabric 0.5 µs | [가정] |
| 최대 outstanding | MOCS 16,384 / 채널 | 무제한 | OCP MOCS CSR [해석] |
| sense unit 수 N | 16(기준 구성) ~ 4096 sweep | 16 bank | OCP §13.1.1 / [외부] |

호스트는 순차(채널 인터리빙) weight stream을 발행하며 `outstanding_bytes + req ≤ W`인 동안만
요청을 낸다. 소비자는 무한히 빠르다고 가정(도착 즉시 슬롯 반환)하므로 결과는 **버퍼의 하한**이다.

## 실행

```
pip install numpy matplotlib
python3 run_experiments.py          # 수 분. results/ 에 CSV·PNG·tables.md 생성
python3 run_experiments.py --quick
```

## 실험

| ID | 내용 |
|---|---|
| E1 | 버퍼 크기 sweep → 달성 BW 곡선. HBM vs HBF(tR 3/20/50 µs), 4 KiB 요청 |
| E2 | 요청 입도 64 B~4 KiB × MOCS on/off → L2 line miss 방식의 한계 |
| E3 | 채널당 sense unit 수 N sweep → 버퍼로 못 푸는 array 상한 |
| E4 | tR jitter → 필요 버퍼 증가분 |
| E5 | 시나리오별 Little's law 요약 (통로 반분 grade 2 포함) |

결과·해석은 `REPORT.md`.
