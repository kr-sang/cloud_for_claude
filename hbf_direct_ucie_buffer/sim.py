"""
Closed-loop latency-hiding simulator for GPU <-> memory-device channels.

Question answered: how many bytes (and how many outstanding requests) must a
GPU keep in flight to run a device at its effective bandwidth, given the
device's access latency?  Compared for HBM and for OCP HBF v0.7.0 attached
directly over UCIe.

Model (per channel; channels are independent in both HBM and HBF, so the
cube/stack result is the per-channel result scaled by the channel count):

  host --cmd--> [device: page cache buffers per bank, sense units (tR)] --data pipe (link BW)--> host

* Host issues sequential (channel-interleaved) reads of `req_bytes` each, but
  only while `outstanding_bytes + req_bytes <= window_bytes` (the buffer that
  must hold data not yet consumed / tags of misses not yet returned).
* Device: page = addr // page_bytes, bank = page % banks (OCP DLU index =
  [page | bank]).  A request hitting a page present in the bank's cache
  buffers (OCP: 2 x 4 KiB per bank by default, NCBB) is served after t_hit.
  Otherwise the bank must sense the page (tR, optionally jittered); a bank
  senses one page at a time (OCP §5.3.1: same-bank strict ordering).  Requests
  for a page that is already being sensed are merged [assumption].
* Data returns through a per-channel pipe of `link_bw` bytes/s (serialised),
  plus a fixed one-way latency each direction.
* Consumer is infinitely fast: a buffer slot is released as soon as the data
  lands.  The result is therefore the *minimum* buffer (lower bound).

All OCP values are from OCP_CONSTRAINTS.md; NAND timings (tR) are not in the
standard and are bracketed.
"""
from __future__ import annotations

import heapq
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field


@dataclass
class ChannelConfig:
    name: str
    link_bw: float            # bytes/s, effective per-channel data bandwidth
    t_cmd: float              # s, one-way host->device command latency
    t_resp: float             # s, one-way device->host data latency (after pipe)
    t_hit: float              # s, page-cache-buffer hit service time
    t_sense: float            # s, tR (HBF) / tRCD+tRP row-miss (HBM)
    sense_jitter: float       # fraction, uniform +/- jitter on t_sense
    page_bytes: int           # 4096 (HBF DLU) / 1024 (HBM3 pseudo-channel page)
    banks: int                # independent sense units per channel
    cache_pages_per_bank: int # OCP NCBB default 2; HBM open row = 1
    max_outstanding: int | None = None  # OCP MOCS, commands per channel
    seed: int = 1


@dataclass
class Result:
    window_bytes: int
    req_bytes: int
    achieved_bw: float        # bytes/s
    mean_latency: float       # s, issue -> data landed
    p99_latency: float
    mean_outstanding_reqs: float
    mean_outstanding_bytes: float
    link_util: float
    sense_util: float


# event kinds
_ARRIVE, _SENSE_DONE, _LANDED = 0, 1, 2


def simulate(cfg: ChannelConfig, window_bytes: int, req_bytes: int,
             warmup: float, measure: float) -> Result:
    rng = random.Random(cfg.seed)
    t_end = warmup + measure
    ev: list = []
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        seq += 1
        heapq.heappush(ev, (t, seq, kind, payload))

    # state
    outstanding_bytes = 0
    outstanding_reqs = 0
    next_addr = 0
    link_free = 0.0
    bank_busy = [False] * cfg.banks
    bank_sensing_page = [-1] * cfg.banks
    bank_queue = [deque() for _ in range(cfg.banks)]            # pages waiting to be sensed (ordered)
    bank_waiting = [dict() for _ in range(cfg.banks)]           # page -> list of (issue_t, bytes)
    bank_cache = [OrderedDict() for _ in range(cfg.banks)]      # page -> True (LRU)

    # accounting (measurement window only)
    landed_bytes = 0
    lat_samples = []
    link_busy_time = 0.0
    sense_busy_time = 0.0
    # time-weighted outstanding
    acc_out_reqs = 0.0
    acc_out_bytes = 0.0
    last_t = 0.0

    def account(t):
        nonlocal acc_out_reqs, acc_out_bytes, last_t
        if t > warmup:
            a = max(last_t, warmup)
            dt = t - a
            if dt > 0:
                acc_out_reqs += outstanding_reqs * dt
                acc_out_bytes += outstanding_bytes * dt
        last_t = t

    def try_issue(t):
        nonlocal outstanding_bytes, outstanding_reqs, next_addr
        while outstanding_bytes + req_bytes <= window_bytes and \
                (cfg.max_outstanding is None or outstanding_reqs < cfg.max_outstanding):
            addr = next_addr
            next_addr += req_bytes
            outstanding_bytes += req_bytes
            outstanding_reqs += 1
            push(t + cfg.t_cmd, _ARRIVE, (t, addr))

    def deliver(t, issue_t, nbytes):
        """page data available at device at time t -> serialise on link pipe."""
        nonlocal link_free, link_busy_time
        start = max(link_free, t)
        xfer = nbytes / cfg.link_bw
        link_free = start + xfer
        if start >= warmup:
            link_busy_time += xfer
        elif link_free > warmup:
            link_busy_time += link_free - warmup
        push(link_free + cfg.t_resp, _LANDED, (issue_t, nbytes))

    def start_sense(t, b):
        nonlocal sense_busy_time
        if bank_busy[b] or not bank_queue[b]:
            return
        page = bank_queue[b].popleft()
        bank_busy[b] = True
        bank_sensing_page[b] = page
        ts = cfg.t_sense * (1 + cfg.sense_jitter * rng.uniform(-1, 1))
        if t >= warmup:
            sense_busy_time += ts
        push(t + ts, _SENSE_DONE, (b, page))

    try_issue(0.0)
    while ev:
        t, _, kind, payload = heapq.heappop(ev)
        if t > t_end:
            break
        account(t)
        if kind == _ARRIVE:
            issue_t, addr = payload
            page = addr // cfg.page_bytes
            b = page % cfg.banks
            cache = bank_cache[b]
            if page in cache:
                cache.move_to_end(page)
                deliver(t + cfg.t_hit, issue_t, req_bytes)
            else:
                w = bank_waiting[b]
                if page in w:
                    w[page].append((issue_t, req_bytes))
                else:
                    w[page] = [(issue_t, req_bytes)]
                    bank_queue[b].append(page)
                    start_sense(t, b)
        elif kind == _SENSE_DONE:
            b, page = payload
            bank_busy[b] = False
            cache = bank_cache[b]
            cache[page] = True
            while len(cache) > cfg.cache_pages_per_bank:
                cache.popitem(last=False)
            for issue_t, nb in bank_waiting[b].pop(page):
                deliver(t, issue_t, nb)
            start_sense(t, b)
        else:  # _LANDED
            issue_t, nb = payload
            outstanding_bytes -= nb
            outstanding_reqs -= 1
            if t >= warmup:
                landed_bytes += nb
                lat_samples.append(t - issue_t)
            try_issue(t)
    account(t_end)

    lat_samples.sort()
    n = len(lat_samples)
    return Result(
        window_bytes=window_bytes,
        req_bytes=req_bytes,
        achieved_bw=landed_bytes / measure,
        mean_latency=sum(lat_samples) / n if n else float("nan"),
        p99_latency=lat_samples[int(0.99 * (n - 1))] if n else float("nan"),
        mean_outstanding_reqs=acc_out_reqs / measure,
        mean_outstanding_bytes=acc_out_bytes / measure,
        link_util=link_busy_time / measure,
        sense_util=sense_busy_time / (measure * cfg.banks),
    )


# ---------------------------------------------------------------------------
# Device configurations (per channel)
# ---------------------------------------------------------------------------
GiB = 1 << 30
MiB = 1 << 20
KiB = 1 << 10
us = 1e-6
ns = 1e-9

# OCP HBF v0.7.0, speed grade 3: 16 channels x 256 GB/s raw, AXI link
# efficiency 0.75 -> 192 GB/s effective per channel, 3072 GB/s per cube.
HBF_CHANNELS = 16
HBF_CH_EFF_BW = 3072e9 / HBF_CHANNELS
HBF_MOCS = 16384  # max outstanding commands (CSR MOCS), per channel [interpretation]


def hbf_channel(tR_us: float, banks: int = 16, jitter: float = 0.0,
                t_hit_us: float = 0.3, t_link_oneway_ns: float = 150.0,
                mocs: int | None = HBF_MOCS, name: str | None = None) -> ChannelConfig:
    """tR, t_hit, link latency are NOT in the standard -> [assumption] brackets.
    banks: sense units per channel. OCP baseline config = 16 banks (die x bank);
    OCP_CONSTRAINTS §10 shows the spec BW needs far more (see N sweep)."""
    return ChannelConfig(
        name=name or f"HBF tR={tR_us:g}us N={banks}",
        link_bw=HBF_CH_EFF_BW,
        t_cmd=t_link_oneway_ns * ns,
        t_resp=t_link_oneway_ns * ns,
        t_hit=t_hit_us * us,
        t_sense=tR_us * us,
        sense_jitter=jitter,
        page_bytes=4 * KiB,
        banks=banks,
        cache_pages_per_bank=2,
        max_outstanding=mocs,
    )


# HBM3e, "6 stacks on a GPU" class (H200: 6 x 0.8 TB/s = 4.8 TB/s).
# Per stack 16 pseudo-channels -> 96 pseudo-channels of 50 GB/s.
HBM_STACKS = 6
HBM_STACK_BW = 0.8e12
HBM_PC_PER_STACK = 16
HBM_CHANNELS = HBM_STACKS * HBM_PC_PER_STACK
HBM_CH_BW = HBM_STACK_BW / HBM_PC_PER_STACK


def hbm_channel(fabric_oneway_ns: float = 250.0, name: str | None = None) -> ChannelConfig:
    """Loaded GPU->HBM round trip ~500-800 ns [external, assumption]:
    fabric/L2-miss path ~250 ns each way + DRAM row hit/miss."""
    return ChannelConfig(
        name=name or f"HBM3e ({2*fabric_oneway_ns:g}ns fabric RT)",
        link_bw=HBM_CH_BW,
        t_cmd=fabric_oneway_ns * ns,
        t_resp=fabric_oneway_ns * ns,
        t_hit=20 * ns,            # CAS-ish row-hit
        t_sense=35 * ns,          # tRCD + tRP class row miss
        sense_jitter=0.0,
        page_bytes=1 * KiB,       # HBM3 pseudo-channel page
        banks=16,                 # 4 bank groups x 4 banks
        cache_pages_per_bank=1,   # open row
        max_outstanding=None,
    )
