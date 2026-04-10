import os
import sys
from pathlib import Path


NUM_CLASS = 20
NUM_OUTPUT_NEURON = 240

# Classifier guards can be tuned without code edits.
SATURATION_SPIKE_RATIO = float(os.environ.get("TB_SATURATION_SPIKE_RATIO", "0.85"))
MIN_INFORMATIVE_FRAMES = int(os.environ.get("TB_MIN_INFORMATIVE_FRAMES", "2"))
MIN_MARGIN_RATIO = float(os.environ.get("TB_MIN_MARGIN_RATIO", "0.25"))
MIN_TOTAL_VOTES = int(os.environ.get("TB_MIN_TOTAL_VOTES", "24"))
MIN_TOP_VOTE = int(os.environ.get("TB_MIN_TOP_VOTE", "6"))
MIN_TOP_SECOND_GAP = int(os.environ.get("TB_MIN_TOP_SECOND_GAP", "2"))
MIN_WINNER_SHARE = float(os.environ.get("TB_MIN_WINNER_SHARE", "0.18"))

def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        matrix = []
        for line in file:
            line = line.strip()
            # Skip empty lines or lines starting with Verilog comments //
            if not line or line.startswith("//"):
                continue
            
            # Now it's safe to convert bits to integers
            row = [int(bit) for bit in line]
            matrix.append(row)
        return matrix

def list_to_binary(matrix):
    binary_str = ''.join(map(str, matrix))
    hex_number = int(binary_str, 2)
    return hex_number
    
def calculate_majority_class(matrix):
    # Accumulate votes from informative frames only.
    total_votes = [0 for _ in range(NUM_CLASS)]
    silent_frames = 0
    saturated_frames = 0
    informative_frames = 0

    for row in matrix:
        row_bits = row[:NUM_OUTPUT_NEURON]
        spike_count = sum(row_bits)
        spike_ratio = spike_count / float(NUM_OUTPUT_NEURON)

        if spike_count == 0:
            silent_frames += 1
            continue

        if spike_ratio >= SATURATION_SPIKE_RATIO:
            saturated_frames += 1
            continue

        informative_frames += 1
        for i, bit in enumerate(row_bits):
            if bit == 1:
                class_index = i % NUM_CLASS
                total_votes[class_index] += 1

    print("\n--- SOFTWARE CLASSIFICATION RESULTS ---")
    print(f"Total Votes (informative frames only):\n{total_votes}")
    print(
        f"Frame stats: informative={informative_frames}, "
        f"silent={silent_frames}, saturated={saturated_frames}, total={len(matrix)}"
    )

    if informative_frames == 1 and silent_frames >= max(1, len(matrix) - 1):
        print(
            "Hint: Detected a first-frame burst followed by near-total silence. "
            "Use TB_RESET_BETWEEN_PICS=0 (or keep TB_HARD_RESET_BETWEEN_PICS=0), "
            "and/or reduce output strictness (e.g., TB_LAYER2_SPIKE_STIM)."
        )

    if informative_frames < MIN_INFORMATIVE_FRAMES:
        print("\nWARNING: Not enough informative frames for a reliable decision.")
        print("Returning UNKNOWN instead of a false class prediction.\n")
        return [-1]

    total_vote_sum = sum(total_votes)
    if total_vote_sum < MIN_TOTAL_VOTES:
        print("\nWARNING: Evidence is too sparse for stable class selection.")
        print(f"Total informative votes={total_vote_sum} < required {MIN_TOTAL_VOTES}")
        print("Returning UNKNOWN instead of a false class prediction.\n")
        return [-1]

    top_class = max(range(NUM_CLASS), key=lambda idx: total_votes[idx])
    sorted_votes = sorted(total_votes, reverse=True)
    top_vote = sorted_votes[0]
    second_vote = sorted_votes[1] if len(sorted_votes) > 1 else 0
    margin_ratio = (top_vote - second_vote) / float(top_vote) if top_vote > 0 else 0.0
    winner_share = (top_vote / float(total_vote_sum)) if total_vote_sum > 0 else 0.0

    if top_vote == 0:
        print("\nWARNING: No informative spikes after filtering.")
        print("Returning UNKNOWN instead of a false class prediction.\n")
        return [-1]

    if margin_ratio < MIN_MARGIN_RATIO:
        print("\nWARNING: Low-confidence class separation.")
        print(
            f"Top vote={top_vote}, second vote={second_vote}, "
            f"margin ratio={margin_ratio:.3f} < {MIN_MARGIN_RATIO:.3f}"
        )
        print("Returning UNKNOWN instead of a false class prediction.\n")
        return [-1]

    if top_vote < MIN_TOP_VOTE:
        print("\nWARNING: Winner vote count is too small.")
        print(f"Top vote={top_vote} < required {MIN_TOP_VOTE}")
        print("Returning UNKNOWN instead of a false class prediction.\n")
        return [-1]

    if (top_vote - second_vote) < MIN_TOP_SECOND_GAP:
        print("\nWARNING: Top-two vote gap is too small.")
        print(
            f"Top vote={top_vote}, second vote={second_vote}, "
            f"gap={top_vote-second_vote} < required {MIN_TOP_SECOND_GAP}"
        )
        print("Returning UNKNOWN instead of a false class prediction.\n")
        return [-1]

    if winner_share < MIN_WINNER_SHARE:
        print("\nWARNING: Winner share is too low.")
        print(
            f"Winner share={winner_share:.3f} < required {MIN_WINNER_SHARE:.3f} "
            f"(top={top_vote}, total={total_vote_sum})"
        )
        print("Returning UNKNOWN instead of a false class prediction.\n")
        return [-1]

    print(
        f"Confidence: top={top_vote}, second={second_vote}, "
        f"gap={top_vote-second_vote}, margin_ratio={margin_ratio:.3f}, "
        f"winner_share={winner_share:.3f}"
    )

    return [top_class]