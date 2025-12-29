"""
# Plot P/D Competition scheduling timeline visualization (V2).

This script visualizes P/D scheduling with N fixed channels (batch size) on y-axis.
Each channel can process multiple requests sequentially over time.

Timeline structure:
- Y-axis: N channels (0 to N-1), representing concurrent processing slots
- X-axis: Time (ms)
- Red bars: Prefill phase
- Blue bars: Decode phase

P/D Scheduling phases:
- Phase 0 (Initial Prefill): All N channels prefill N requests simultaneously
- Phase 1 (Decode): All N channels decode, until k requests complete
- Phase 2 (Refill Prefill): k completed channels prefill new requests,
                            remaining N-k channels wait
- Back to Phase 1: All N channels decode together
- Repeat...

Usage:
    # From saved timeline data (from kstar_vs_khat_real_model.py)
    python experiments/plot_pd_timeline_v2.py \
        --timeline-json experiments/results/timeline_kstar.json \
        --output-dir experiments/timeline_plots

    # Or simulate based on parameters
    python experiments/plot_pd_timeline_v2.py \
        --simulate \
        --num-requests 200 \
        --batch-size 64 \
        --k-star 15 \
        --output-dir experiments/timeline_plots
"""

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ChannelEvent:
    """Event on a channel."""
    channel_id: int
    request_id: int
    event_type: str  # 'prefill' or 'decode'
    start_time: float  # ms
    end_time: float  # ms


def simulate_pd_timeline_channels(
    num_requests: int,
    N: int,
    k: int,
    avg_prefill_time: float = 50.0,
    avg_decode_time: float = 200.0,
    prefill_variance: float = 0.2,
    decode_variance: float = 0.3,
) -> tuple[list[ChannelEvent], list[tuple[float, str]], float]:
    """
    Simulate P/D scheduling timeline with N channels.

    Args:
        num_requests: Total number of requests to process
        N: Batch size (number of channels)
        k: Switching threshold
        avg_prefill_time: Average prefill time per request (ms)
        avg_decode_time: Average decode time per request (ms)
        prefill_variance: Variance factor for prefill time
        decode_variance: Variance factor for decode time

    Returns:
        events: List of ChannelEvent objects
        phase_transitions: List of (time, phase_name) tuples
        total_time: Total processing time
    """
    np.random.seed(42)  # For reproducibility

    events = []
    phase_transitions = []
    current_time = 0.0

    # Generate random prefill and decode times for each request
    prefill_times = np.random.normal(
        avg_prefill_time,
        avg_prefill_time * prefill_variance,
        num_requests
    )
    prefill_times = np.maximum(prefill_times, avg_prefill_time * 0.3)

    decode_times = np.random.normal(
        avg_decode_time,
        avg_decode_time * decode_variance,
        num_requests
    )
    decode_times = np.maximum(decode_times, avg_decode_time * 0.3)

    # Track channel state
    # channel_request[i] = current request being processed on channel i
    channel_request = [-1] * N
    # channel_decode_end[i] = when decode ends for current request on channel i
    channel_decode_end = [0.0] * N

    next_request_id = 0
    completed_requests = 0

    # Phase 0: Initial Prefill - all N channels prefill simultaneously
    phase_transitions.append((current_time, "INITIAL_PREFILL"))

    initial_batch = min(N, num_requests)
    max_prefill_end = current_time

    for ch in range(initial_batch):
        req_id = next_request_id
        next_request_id += 1
        channel_request[ch] = req_id

        prefill_start = current_time
        prefill_end = prefill_start + prefill_times[req_id]

        events.append(ChannelEvent(
            channel_id=ch,
            request_id=req_id,
            event_type='prefill',
            start_time=prefill_start,
            end_time=prefill_end
        ))

        max_prefill_end = max(max_prefill_end, prefill_end)

    # All prefills complete, start decode
    current_time = max_prefill_end

    # Initialize decode end times
    for ch in range(initial_batch):
        req_id = channel_request[ch]
        decode_start = current_time
        decode_end = decode_start + decode_times[req_id]
        channel_decode_end[ch] = decode_end

        events.append(ChannelEvent(
            channel_id=ch,
            request_id=req_id,
            event_type='decode',
            start_time=decode_start,
            end_time=decode_end
        ))

    phase_transitions.append((current_time, "DECODE"))

    # Main loop: process remaining requests
    while completed_requests < num_requests:
        # Find the k channels that will complete first
        active_channels = [
            (ch, channel_decode_end[ch])
            for ch in range(N)
            if channel_request[ch] >= 0 and channel_decode_end[ch] > current_time
        ]

        if not active_channels:
            # All current requests are done
            break

        # Sort by decode end time
        active_channels.sort(key=lambda x: x[1])

        # Wait for k completions (or all remaining if less than k)
        k_to_complete = min(k, len(active_channels))

        # Time when k-th completion happens
        k_completion_time = active_channels[k_to_complete - 1][1]

        # Channels that complete by k_completion_time
        completed_channels = [
            ch for ch, end_time in active_channels
            if end_time <= k_completion_time
        ]

        # Update completed count
        completed_requests += len(completed_channels)
        current_time = k_completion_time

        # Check if we have more requests to process
        remaining_requests = num_requests - next_request_id

        if remaining_requests <= 0:
            # No more requests, just let remaining decode finish
            if active_channels:
                final_time = max(end_time for _, end_time in active_channels)
                current_time = final_time
            break

        # Phase 2: Refill Prefill - prefill new requests on completed channels
        num_to_prefill = min(len(completed_channels), remaining_requests)

        if num_to_prefill > 0:
            phase_transitions.append((current_time, "REFILL_PREFILL"))

            prefill_start = current_time
            max_prefill_end = current_time

            channels_to_refill = completed_channels[:num_to_prefill]

            for ch in channels_to_refill:
                req_id = next_request_id
                next_request_id += 1
                channel_request[ch] = req_id

                prefill_end = prefill_start + prefill_times[req_id]

                events.append(ChannelEvent(
                    channel_id=ch,
                    request_id=req_id,
                    event_type='prefill',
                    start_time=prefill_start,
                    end_time=prefill_end
                ))

                max_prefill_end = max(max_prefill_end, prefill_end)

            # Wait for all prefills to complete
            current_time = max_prefill_end

            # Start decode for newly prefilled requests
            phase_transitions.append((current_time, "DECODE"))

            for ch in channels_to_refill:
                req_id = channel_request[ch]
                decode_start = current_time
                decode_end = decode_start + decode_times[req_id]
                channel_decode_end[ch] = decode_end

                events.append(ChannelEvent(
                    channel_id=ch,
                    request_id=req_id,
                    event_type='decode',
                    start_time=decode_start,
                    end_time=decode_end
                ))

            # Also resume decode for channels that were waiting
            for ch, end_time in active_channels:
                if ch not in completed_channels and end_time > current_time:
                    # This channel was still decoding, continue
                    pass

    # Find total time
    total_time = max(
        (e.end_time for e in events if e.event_type == 'decode'),
        default=current_time
    )

    return events, phase_transitions, total_time


def plot_timeline_channels(
    events: list[ChannelEvent],
    phase_transitions: list[tuple[float, str]],
    total_time: float,
    N: int,
    k: int,
    policy_name: str,
    output_path: str,
    show_request_ids: bool = False
):
    """
    Plot the P/D scheduling timeline with N channels on y-axis.

    Args:
        events: List of ChannelEvent objects
        phase_transitions: List of (time, phase_name) tuples
        total_time: Total processing time
        N: Batch size (number of channels)
        k: Switching threshold
        policy_name: Name for the plot title
        output_path: Path to save the plot
        show_request_ids: Whether to show request IDs on bars
    """
    _, ax = plt.subplots(figsize=(16, 10))

    # Plot each event
    for event in events:
        y = event.channel_id
        width = event.end_time - event.start_time

        color = '#E74C3C' if event.event_type == 'prefill' else '#3498DB'

        ax.barh(y, width, left=event.start_time,
                height=0.8, color=color, edgecolor='white',
                linewidth=0.5, alpha=0.9)

        # Optionally show request ID
        if show_request_ids and width > total_time * 0.02:
            ax.text(event.start_time + width / 2, y,
                    str(event.request_id),
                    ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold')

    # Add vertical lines for phase transitions
    for trans_time, phase in phase_transitions:
        if 'PREFILL' in phase:
            ax.axvline(x=trans_time, color='red', linestyle='--',
                       alpha=0.4, linewidth=1)

    # Styling
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Channel ID', fontsize=14)
    ax.set_title(f'P/D Scheduling Timeline\n{policy_name} (k={k}, N={N})',
                 fontsize=16, fontweight='bold')

    ax.set_xlim(0, total_time * 1.02)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_yticks(range(N))

    # Legend
    prefill_patch = mpatches.Patch(color='#E74C3C', label='Prefill')
    decode_patch = mpatches.Patch(color='#3498DB', label='Decode')
    ax.legend(handles=[prefill_patch, decode_patch],
              loc='upper right', fontsize=12, framealpha=0.9)

    ax.ticklabel_format(style='plain', axis='x')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def load_timeline_from_json(
    json_path: str
) -> tuple[list[ChannelEvent], list, float, int, int]:
    """
    Load timeline data from JSON file.

    Expected format:
    {
        "k": int,
        "N": int,
        "total_time_ms": float,
        "events": [
            {"channel_id": int, "request_id": int, "event_type": str,
             "start_time": float, "end_time": float},
            ...
        ],
        "phase_transitions": [[time, phase_name], ...]
    }
    """
    with open(json_path) as f:
        data = json.load(f)

    events = []
    for e in data.get('events', []):
        events.append(ChannelEvent(
            channel_id=e['channel_id'],
            request_id=e['request_id'],
            event_type=e['event_type'],
            start_time=e['start_time'],
            end_time=e['end_time']
        ))

    phase_transitions = [
        (t, p) for t, p in data.get('phase_transitions', [])
    ]

    total_time = data.get('total_time_ms', 0)
    k = data.get('k', 1)
    N = data.get('N', 64)

    return events, phase_transitions, total_time, k, N


def compute_analytical_kstar(N: int, p: float, alpha_p: float,
                             alpha_d: float, beta_d: float) -> int:
    """Compute optimal k* using Proposition 1."""
    def compute_tau(batch_size: int) -> float:
        if batch_size <= 0:
            return float('inf')
        numerator = alpha_d + beta_d * batch_size
        denominator = 1.0 - (1.0 - p) ** batch_size
        if denominator <= 1e-10:
            return float('inf')
        return numerator / denominator

    for k in range(1, N + 1):
        lhs = k * compute_tau(N - k)
        sum_tau = sum(compute_tau(j) for j in range(N - k + 1, N + 1))
        rhs = sum_tau + alpha_p

        if lhs >= rhs:
            return max(1, k)

    return max(1, N // 5)


def main():
    parser = argparse.ArgumentParser(
        description="Plot P/D scheduling timeline with N channels"
    )
    parser.add_argument("--timeline-json", type=str, default=None,
                        help="Path to timeline JSON file from experiment")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate timeline instead of loading from file")
    parser.add_argument("--num-requests", type=int, default=200,
                        help="Number of requests (for simulation)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size N (for simulation)")
    parser.add_argument("--k-star", type=int, default=None,
                        help="k* value (for simulation)")
    parser.add_argument("--k-hat", type=int, default=None,
                        help="k_hat value (for simulation)")
    parser.add_argument("--avg-prefill-time", type=float, default=50.0,
                        help="Average prefill time in ms (for simulation)")
    parser.add_argument("--avg-decode-time", type=float, default=200.0,
                        help="Average decode time in ms (for simulation)")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/timeline_plots")
    parser.add_argument("--show-request-ids", action="store_true",
                        help="Show request IDs on bars")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.timeline_json:
        # Load from JSON file
        print(f"Loading timeline from {args.timeline_json}...")
        events, phase_transitions, total_time, k, N = load_timeline_from_json(
            args.timeline_json
        )

        policy_name = os.path.basename(args.timeline_json).replace('.json', '')

        plot_timeline_channels(
            events, phase_transitions, total_time, N, k,
            policy_name=policy_name,
            output_path=os.path.join(args.output_dir, f"{policy_name}.png"),
            show_request_ids=args.show_request_ids
        )

    elif args.simulate:
        # Simulate timelines
        N = args.batch_size
        num_requests = args.num_requests

        # Compute k values
        alpha_p, alpha_d, beta_d, p = 0.01, 0.001, 0.0001, 0.01
        k_star = args.k_star or compute_analytical_kstar(
            N, p, alpha_p, alpha_d, beta_d
        )
        k_hat = args.k_hat or N // 2

        print(f"Simulating P/D timelines with N={N}, k*={k_star}, k_hat={k_hat}")
        print(f"Number of requests: {num_requests}")

        # Simulate k* policy
        print(f"\nSimulating k* policy (k={k_star})...")
        events_kstar, trans_kstar, time_kstar = simulate_pd_timeline_channels(
            num_requests, N, k_star,
            avg_prefill_time=args.avg_prefill_time,
            avg_decode_time=args.avg_decode_time
        )

        plot_timeline_channels(
            events_kstar, trans_kstar, time_kstar, N, k_star,
            policy_name="k* policy",
            output_path=os.path.join(args.output_dir, f"timeline_kstar_k{k_star}.png"),
            show_request_ids=args.show_request_ids
        )

        # Save k* timeline data
        kstar_data = {
            'policy': 'k*',
            'k': k_star,
            'N': N,
            'num_requests': num_requests,
            'total_time_ms': time_kstar,
            'events': [
                {
                    'channel_id': e.channel_id,
                    'request_id': e.request_id,
                    'event_type': e.event_type,
                    'start_time': e.start_time,
                    'end_time': e.end_time
                }
                for e in events_kstar
            ],
            'phase_transitions': trans_kstar
        }
        kstar_json_path = os.path.join(
            args.output_dir, f"timeline_kstar_k{k_star}.json"
        )
        with open(kstar_json_path, 'w') as f:
            json.dump(kstar_data, f, indent=2)
        print(f"Saved: {kstar_json_path}")

        # Simulate k_hat policy
        print(f"\nSimulating k_hat policy (k={k_hat})...")
        events_khat, trans_khat, time_khat = simulate_pd_timeline_channels(
            num_requests, N, k_hat,
            avg_prefill_time=args.avg_prefill_time,
            avg_decode_time=args.avg_decode_time
        )

        plot_timeline_channels(
            events_khat, trans_khat, time_khat, N, k_hat,
            policy_name="k_hat policy",
            output_path=os.path.join(args.output_dir, f"timeline_khat_k{k_hat}.png"),
            show_request_ids=args.show_request_ids
        )

        # Save k_hat timeline data
        khat_data = {
            'policy': 'k_hat',
            'k': k_hat,
            'N': N,
            'num_requests': num_requests,
            'total_time_ms': time_khat,
            'events': [
                {
                    'channel_id': e.channel_id,
                    'request_id': e.request_id,
                    'event_type': e.event_type,
                    'start_time': e.start_time,
                    'end_time': e.end_time
                }
                for e in events_khat
            ],
            'phase_transitions': trans_khat
        }
        khat_json_path = os.path.join(args.output_dir, f"timeline_khat_k{k_hat}.json")
        with open(khat_json_path, 'w') as f:
            json.dump(khat_data, f, indent=2)
        print(f"Saved: {khat_json_path}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"k* policy (k={k_star}): {time_kstar:.0f} ms")
        print(f"k_hat policy (k={k_hat}): {time_khat:.0f} ms")
        print(f"Throughput k*: {num_requests / (time_kstar / 1000):.2f} req/s")
        print(f"Throughput k_hat: {num_requests / (time_khat / 1000):.2f} req/s")
        print(f"Plots saved to {args.output_dir}/")

    else:
        parser.print_help()
        print("\nError: Either --timeline-json or --simulate is required")


if __name__ == "__main__":
    main()
