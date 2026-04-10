import os
import sys
from pathlib import Path

import cocotb
from rram_neuron_model import conductance_at_age
# from cocotb.binary import BinaryRepresentation, BinaryValue
from cocotb.triggers import Timer
from cocotb.clock import Clock
from cocotb.handle import SimHandleBase
from cocotb.queue import Queue
# from cocotb.runner import get_runner
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
from nvm_parameter import *
from read_file import *

async def load_and_inject_stimuli(dut, file_path):
    # 1. Read the text file we generated
    with open(file_path, "r") as f:
        # Filter out comments and whitespace
        spike_data = [line.strip() for line in f if line.strip() and not line.startswith("//")]

    dut._log.info(f"Loaded {len(spike_data)} rows of spikes from {file_path}")

    # 2. Feed it into the SNN row-by-row
    # Assuming your SNN is mapped to the 'mprj' pins or internal wires
    for i, binary_row in enumerate(spike_data):
        # Convert "0010..." to integer
        spike_val = int(binary_row, 2)
        
        # Drive the value into your user project
        # Update 'uut.chip_core.mprj' to match your actual SNN input port name
        dut.uut.chip_core.mprj.user_project.input_spikes.value = spike_val
        
        # Wait for 1 clock cycle (or match your Ramp modulation timing)
        await ClockCycles(dut.clk, 1)
        
        if i % 32 == 0:
            dut._log.info(f"Processing Frame {i//32}...")

TB_DIR = Path(__file__).resolve().parent
BASE_DIR = TB_DIR / "mem" / "connection"

_default_input = TB_DIR / "stimuli_class2.txt"
_fallback_input = TB_DIR / "mem" / "stimuli" / "stimuli.txt"
INPUT_FILE = Path(os.environ.get("TB_STIMULI_FILE", str(_default_input)))
if not INPUT_FILE.exists() and _fallback_input.exists():
    INPUT_FILE = _fallback_input

LAYER1_SPIKE_STIM = int(os.environ.get("TB_LAYER1_SPIKE_STIM", "1"))
LAYER2_SPIKE_STIM = int(os.environ.get("TB_LAYER2_SPIKE_STIM", "4"))
DEBUG_ACTIVE_AXON = os.environ.get("TB_DEBUG_ACTIVE_AXON", "0") == "1"
L0_STIM_MODE = os.environ.get("TB_L0_STIM_MODE", "raw16").strip().lower()
L0_STIM_GAIN = int(os.environ.get("TB_L0_STIM_GAIN", "1"))
INTER_PIC_IDLE_CYCLES = int(os.environ.get("TB_INTER_PIC_IDLE_CYCLES", "0"))
RESET_BETWEEN_PICS = os.environ.get("TB_RESET_BETWEEN_PICS", "0") == "1"
HARD_RESET_BETWEEN_PICS = os.environ.get("TB_HARD_RESET_BETWEEN_PICS", "0") == "1"

RRAM_AGE_S = float(os.environ.get("TB_RRAM_AGE_S", "1500.0"))
RRAM_LRS_KEEP_THRESHOLD = float(os.environ.get("TB_RRAM_LRS_KEEP_THRESHOLD", "1.5"))
DISABLE_RRAM_DEGRADATION = os.environ.get("TB_DISABLE_RRAM_DEGRADATION", "0") == "1"


def load_stimuli_frames(file_path, requested_pics=None):
    """Load 32-bit row stimuli grouped as 32 rows per frame."""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("//")]

    if len(lines) < 32:
        raise ValueError(f"Stimuli file has only {len(lines)} valid rows: {file_path}")

    available_pics = len(lines) // 32
    if requested_pics is None:
        pics_to_use = available_pics
    else:
        pics_to_use = min(requested_pics, available_pics)

    stimuli = []
    for pic in range(pics_to_use):
        frame_data = []
        for row in range(32):
            row_int = int(lines[pic * 32 + row], 2)
            frame_data.append(row_int)
        stimuli.append(frame_data)

    return stimuli


def encode_l0_block(block_16b):
    """Encode a 16-bit input block into the scalar stimulus consumed by the neuron core."""
    mode = L0_STIM_MODE
    if mode == "raw16":
        return block_16b

    bit_count = bin(block_16b & 0xFFFF).count("1")
    if mode == "popcount16":
        return bit_count * max(1, L0_STIM_GAIN)
    if mode == "binary_presence":
        return max(1, L0_STIM_GAIN) if bit_count > 0 else 0

    raise ValueError(f"Unsupported TB_L0_STIM_MODE={L0_STIM_MODE}. Use raw16|popcount16|binary_presence")


def summarize_stimuli_activity(stimuli):
    if not stimuli:
        print("Stimuli summary: no frames")
        return
    frame_ones = []
    for frame in stimuli:
        ones = sum(bin(row).count("1") for row in frame)
        frame_ones.append(ones)
    print(
        "Stimuli summary: "
        f"frames={len(stimuli)}, "
        f"frame_ones_min={min(frame_ones)}, "
        f"frame_ones_max={max(frame_ones)}, "
        f"frame_ones_avg={sum(frame_ones)/float(len(frame_ones)):.2f}"
    )


def summarize_layer_activity(layer_name, spike_matrix):
    """Print compact spike statistics to detect silence/saturation quickly."""
    if not spike_matrix:
        print(f"{layer_name} activity: no frames")
        return

    frame_count = len(spike_matrix)
    neuron_count = len(spike_matrix[0]) if spike_matrix[0] else 0
    total_slots = frame_count * neuron_count
    total_spikes = sum(sum(frame) for frame in spike_matrix)
    spike_ratio = (total_spikes / float(total_slots)) if total_slots > 0 else 0.0
    per_frame_counts = [sum(frame) for frame in spike_matrix]
    unique_frame_count = len({tuple(frame) for frame in spike_matrix})
    min_frame_spikes = min(per_frame_counts) if per_frame_counts else 0
    max_frame_spikes = max(per_frame_counts) if per_frame_counts else 0

    print(
        f"{layer_name} activity summary: total_spikes={total_spikes}, "
        f"spike_ratio={spike_ratio:.4f}, frame_min={min_frame_spikes}, "
        f"frame_max={max_frame_spikes}, unique_frames={unique_frame_count}/{frame_count}"
    )
# --- Helper Functions for Wishbone and NVM Access ---

# Wishbone Write: Used to send control or configuration data to the DUT.
async def wishbone_write(dut, address, data):
    """
    Performs a Wishbone write transaction.
    
    The transaction is:
    1. Drive signals on positive clock edge.
    2. De-assert signals on the next positive clock edge.
    3. Wait for one falling edge (optional, often used for cycle completion).
    """
    # Cycle 1: Assert request signals
    await RisingEdge(dut.wb_clk_i)
    dut.wbs_cyc_i.value = 1       # Cycle valid
    dut.wbs_stb_i.value = 1       # Strobe (data valid)
    dut.wbs_we_i.value = 1        # Write enable
    dut.wbs_sel_i.value = 0b1111  # Byte select (all bytes enabled for 32-bit write)
    dut.wbs_adr_i.value = address # Address
    dut.wbs_dat_i.value = data    # Write data

    # Cycle 2: De-assert request signals
    await RisingEdge(dut.wb_clk_i)
    dut.wbs_cyc_i.value = 0
    dut.wbs_stb_i.value = 0
    dut.wbs_we_i.value = 0
    dut.wbs_sel_i.value = 0b0000

    # Wait for completion (often necessary depending on DUT design)
    await FallingEdge(dut.wb_clk_i)

# Wishbone Read: Used for reading spikes out or output packets from the last core.
async def wishbone_read(dut, address, spike_o_matrix=None, pic=0, slice_idx=0, layer=0, core=0):
    """
    Performs a Wishbone read transaction.
    
    The transaction is:
    1. Assert request signals (read enable = 0) on positive clock edge.
    2. De-assert signals on the next positive clock edge.
    3. Read data on the falling edge after the de-assertion.
    
    If spike_o_matrix is provided, it extracts spike data from the read value.
    """
    # Cycle 1: Assert request signals
    await RisingEdge(dut.wb_clk_i)
    dut.wbs_cyc_i.value = 1       # Cycle valid
    dut.wbs_stb_i.value = 1       # Strobe (data valid)
    dut.wbs_we_i.value = 0        # Read enable (Write enable is 0)
    dut.wbs_sel_i.value = 0b1111  # Byte select
    dut.wbs_adr_i.value = address # Address
    dut.wbs_dat_i.value = 0       # Don't care for read

    # Cycle 2: De-assert request signals
    await RisingEdge(dut.wb_clk_i)
    dut.wbs_cyc_i.value = 0
    dut.wbs_stb_i.value = 0
    dut.wbs_sel_i.value = 0
        
    # Wait for data to be stable (assuming data is valid after de-assertion)
    await FallingEdge(dut.wb_clk_i)
    
    # Read the output data from the DUT
    # The output spike is expected to be reversed (LSB first) for easier indexing.
    # Assumes dut.wbs_dat_o.value returns a BinaryValue
    output_spike = str(dut.wbs_dat_o.value)[::-1]
    
    # If a spike matrix is provided, parse and store the spike outputs
    if spike_o_matrix is not None:
        # NUM_NEURON_PER_SLICE must be the number of bits in output_spike
        for i in range(NUM_NEURON_PER_SLICE):
            # Calculate the global neuron index in the layer's output matrix
            global_neuron_index = core * NUM_NEURON + slice_idx * NUM_NEURON_PER_SLICE + i
            # Store the spike (0 or 1)
            spike_o_matrix[pic][global_neuron_index] = int(output_spike[i])

# NVM Write: Performs a Wishbone write with an additional delay for Non-Volatile Memory (NVM) programming.
async def nvm_write(dut, address, data):
    """
    Performs a Write operation to the NVM block, incorporating the NVM programming delay.
    """
    async def drive_wishbone():
        # Wishbone transaction setup
        await RisingEdge(dut.wb_clk_i)
        dut.wbs_cyc_i.value = 1
        dut.wbs_stb_i.value = 1
        dut.wbs_we_i.value = 1
        dut.wbs_sel_i.value = 0b1111
        dut.wbs_adr_i.value = address
        dut.wbs_dat_i.value = data

        await RisingEdge(dut.wb_clk_i)
        dut.wbs_cyc_i.value = 0
        dut.wbs_stb_i.value = 0
        dut.wbs_we_i.value = 0
        dut.wbs_sel_i.value = 0

    async def wait_for_delay():
        # Wait for the NVM Write Delay (WR_Dly) to ensure programming completes
        await ClockCycles(dut.wb_clk_i, (2 * WR_Dly + 1))

    # Start the Wishbone drive and the delay concurrently
    drive_task = cocotb.start_soon(drive_wishbone())
    delay_task = cocotb.start_soon(wait_for_delay())

    # Wait for both tasks to complete (drive_task finishes quickly, delay_task takes time)
    await drive_task
    await delay_task

# NVM Read: Performs a pseudo-write (for command/stimulus) followed by an NVM Read operation.
async def nvm_read(dut, addr, data):
    """
    Performs a Read operation from the NVM block, consisting of:
    1. A 'Write' phase to configure the address/stimulus (no NVM write delay needed here).
    2. A delay for NVM Read access time (RD_Dly).
    3. A subsequent 'Read' phase to get the data.
    """
    async def operation_1_write():
        # Phase 1: 'Write' operation to set up command/stimulus (like a configuration write)
        await RisingEdge(dut.wb_clk_i)
        dut.wbs_cyc_i.value = 1
        dut.wbs_stb_i.value = 1
        dut.wbs_we_i.value = 1
        dut.wbs_sel_i.value = 0xF
        dut.wbs_adr_i.value = addr
        dut.wbs_dat_i.value = data

        await RisingEdge(dut.wb_clk_i)
        dut.wbs_cyc_i.value = 0
        dut.wbs_stb_i.value = 0
        dut.wbs_we_i.value  = 0
        dut.wbs_sel_i.value = 0

    async def operation_2_read_after_delay():
        # Wait for the NVM Read Delay (RD_Dly)
        await ClockCycles(dut.wb_clk_i, (RD_Dly + 2))

        # Phase 2: 'Read' operation
        await RisingEdge(dut.wb_clk_i)
        dut.wbs_cyc_i.value = 1
        dut.wbs_stb_i.value = 1
        dut.wbs_we_i.value  = 0 # Read: we_i = 0
        dut.wbs_sel_i.value = 0xF
        dut.wbs_adr_i.value = addr # Re-assert address
        # dut.wbs_dat_i.value is don't care for read

        await RisingEdge(dut.wb_clk_i)
        dut.wbs_cyc_i.value = 0
        dut.wbs_stb_i.value = 0
        dut.wbs_sel_i.value = 0

    # Start the two operations sequentially (task_1 must finish before task_2 can proceed past the delay)
    task_1 = cocotb.start_soon(operation_1_write())
    await task_1 
    
    task_2 = cocotb.start_soon(operation_2_read_after_delay())
    await task_2

# --- Testbench Functions ---

def get_connection_file_path(base_dir, index, part=None):
    """Constructs the full path for a connection file."""
    if part:
        filename = f"connection_{index:03}_part{part}.txt"
    else:
        filename = f"connection_{index:03}.txt"
    return str(Path(base_dir) / filename)

def load_connection_matrices(base_path):
    """Loads all connection matrices into a dictionary."""
    print("Loading connection files...")
    connection_matrices = {}
        
    # Layer 0 connections (0 to 12)
    for i in range(13):
        path = get_connection_file_path(base_path, i)
        connection_matrices[i] = read_matrix_from_file(path)
        
    # Layer 1 connections (13 to 16)
    for i in range(13, 17):
        path = get_connection_file_path(base_path, i)
        connection_matrices[i] = read_matrix_from_file(path)
        
    # Layer 2 connection (index 26 parts 1 to 4)
    for part in range(1, 5):
        index = 26 + part / 10 # Using float key temporarily
        path = get_connection_file_path(base_path, 26, part)
        connection_matrices[index] = read_matrix_from_file(path)
        
    return connection_matrices

async def program_layer_connections(dut, core_idx, layer_conn, NUM_NEURON_LAYER):
    for row_i in range(32):
        for col_i in range(32):
            row = row_i
            col = col_i
            
            axon_group = (row & 0x07) * 32 
            axon = axon_group + col 
            
            neuron_index_group = (row >> 3) & 0x03
            neuron = neuron_index_group * 16 
            
            # Get the ideal, perfect bits from the text file
            # Get the ideal, perfect bits from the text file
            val_slice = layer_conn[axon][NUM_NEURON_LAYER - (neuron + 16):NUM_NEURON_LAYER - neuron] 
            
            degraded_slice = []
            for bit in val_slice:
                if bit == 1:
                    if DISABLE_RRAM_DEGRADATION:
                        degraded_bit = 1
                    else:
                        # Calculate true LRS (Low Resistance State) conductance.
                        g_val = conductance_at_age(RRAM_AGE_S, state="LRS", modulation="ramp")
                        # If conductance drops too low, the digital bit flips to 0.
                        degraded_bit = 1 if g_val > RRAM_LRS_KEEP_THRESHOLD else 0
                else:
                    degraded_bit = 0
                    
                degraded_slice.append(degraded_bit)

            # Convert the physically-degraded bits to an integer for the Verilog chip
            int_val = list_to_binary(degraded_slice)

            data_to_write = (
                (MODE_PROGRAM << 30) |
                (row          << 25) |
                (col          << 20) |
                (0            << 16) |
                int_val
            )

            await nvm_write(dut, 0x30000000, data_to_write)
            
async def run_layer_for_all_pics(dut, core_idx, layer, num_cores, spike_in_matrix, spike_out_matrix, num_pics, stimuli=None, layer_axon_limit=None):
    """
    Runs the simulation for all input pictures for a specific core (EVERY PIC step).
    
    This corresponds to the 'MODE_READ' operation (stimulus application).
    """
    per_pic_active_axons = []
    per_pic_input_signature = []

    for pic in range(num_pics):
        print(f"Layer {layer} - Core {core_idx} - Pic {pic}")

        # Optional frame separation to avoid carry-over dynamics from previous picture.
        if pic > 0 and INTER_PIC_IDLE_CYCLES > 0:
            await ClockCycles(dut.wb_clk_i, INTER_PIC_IDLE_CYCLES)
        if pic > 0 and RESET_BETWEEN_PICS:
            # Soft clear between pictures: preserve programmed weights, clear per-slice state.
            await wishbone_write(dut, 0x30002000, 0)
            await wishbone_write(dut, 0x30002002, 0)
            await wishbone_write(dut, 0x30002004, 0)
            await wishbone_write(dut, 0x30002006, 0)

            # Optional hard reset is explicit because it can wipe programmed state.
            if HARD_RESET_BETWEEN_PICS:
                dut.wb_rst_i.value = 1
                await ClockCycles(dut.wb_clk_i, 2)
                dut.wb_rst_i.value = 0

        active_axon_count = 0
        input_signature = 0
        
        # Iterate over the NVM array structure
        for row_i in range(32):
            for col_i in range(32):
                row = row_i
                col = col_i
                
                # Axon index calculation
                axon = ((row & 0x07) * 32) + col 
                
                # Check for axon limit specific to the layer (e.g., if the layer has fewer than 256 inputs)
                if layer_axon_limit is not None and axon >= layer_axon_limit:
                    continue 
                
                # Neuron index calculation (not used directly here, but for completeness)
                # neuron = ((row >> 3) & 0x03) * 16 
                
                spike_active = False
                val_slice = 0
                
                # Input stimuli (spike_in) depends on the layer:
                if layer == 0:
                    # Layer 0: The stimuli is read directly from our text file
                    # We need to take the 32-bit row and split it into two 16-bit blocks
                    # based on whether it is an even or odd axon group.
                    
                    full_stimuli_val = stimuli[pic][row] 
                    
                    if (axon % 2) == 0:
                        # Even axon: Take the upper 16 bits [31:16]
                        block_16b = (full_stimuli_val >> 16) & 0xFFFF
                    else:
                        # Odd axon: Take the lower 16 bits [15:0]
                        block_16b = full_stimuli_val & 0xFFFF

                    val_slice = encode_l0_block(block_16b)
                        
                    spike_active = True # We always trigger the read for L0 to inject the block
                elif layer == 1 and spike_in_matrix is not None:
                    input_idx = core_idx * NUM_AXON_LAYER_1 + axon
                    if input_idx < len(spike_in_matrix[pic]) and spike_in_matrix[pic][input_idx] == 1:
                        val_slice = max(0, min(0xFFFF, LAYER1_SPIKE_STIM))
                        spike_active = True
                elif layer == 2 and spike_in_matrix is not None:
                    input_idx = axon
                    if input_idx < len(spike_in_matrix[pic]) and spike_in_matrix[pic][input_idx] == 1:
                        val_slice = max(0, min(0xFFFF, LAYER2_SPIKE_STIM))
                        spike_active = True
                
                
                if spike_active:
                    active_axon_count += 1
                    input_signature = ((input_signature * 1315423911) ^ (axon + 0x9E3779B9)) & 0xFFFFFFFF
                    # Construct the 32-bit data word for NVM Read Operation (stimulus application)
                    # {MODE_READ(2), row(5), col(5), padding(4), stimulus_data(16)}
                    data_for_read_op = (
                        (MODE_READ << 30) |  # 2 MSBs for MODE
                        (row         << 25) |  # 5 bits for row index
                        (col         << 20) |  # 5 bits for column index
                        (0           << 16) |  # 4 bits padding (0)
                        (val_slice)             # 16 LSBs for stimulus data (or just '1' for spike)
                    )
                    
                    await nvm_read(dut, 0x30000000, data_for_read_op)

                # Control Register Writes (e.g., resetting internal slice/neuron logic after an NVM row sweep)
                # These addresses (0x3000200x) likely correspond to control registers for neuron slices.
                # The writes happen after every NVM array row group (e.g., after rows 7, 15, 23, 31)
                if col == 31:
                    if row == 7:
                        await wishbone_write(dut, 0x30002000, 0)
                    elif row == 15:
                        await wishbone_write(dut, 0x30002002, 0)
                    elif row == 23:
                        await wishbone_write(dut, 0x30002004, 0)
                    elif row == 31:
                        await wishbone_write(dut, 0x30002006, 0)

        # Read the output spikes from the two slices of the core after processing all inputs
        await wishbone_read(dut, 0x30001000, spike_out_matrix, pic, slice_idx=0, layer=layer, core=core_idx)
        await wishbone_read(dut, 0x30001004, spike_out_matrix, pic, slice_idx=1, layer=layer, core=core_idx) 
        per_pic_active_axons.append(active_axon_count)
        per_pic_input_signature.append(input_signature)
        if DEBUG_ACTIVE_AXON and layer in (1, 2):
            print(
                f"L{layer} Core {core_idx} Pic {pic} "
                f"active_axons={active_axon_count} sig=0x{input_signature:08X}"
            )

    if layer in (1, 2) and per_pic_active_axons:
        min_active = min(per_pic_active_axons)
        max_active = max(per_pic_active_axons)
        unique_sig = len(set(per_pic_input_signature))
        print(
            f"L{layer} Core {core_idx} input summary: "
            f"active_axons[min,max]=[{min_active},{max_active}], "
            f"unique_signatures={unique_sig}/{len(per_pic_input_signature)}"
        )

# --- Cocotb Test ---

@cocotb.test()
async def neuron_network_test(dut): 
    # Determine the base directory for input files (assuming a fixed structure)
    base_dir = BASE_DIR
    
    # Load all connection matrices efficiently
    connection_matrices = load_connection_matrices(base_dir)
    
    # Load stimuli and correct output files
    # TB_NUM_PICS can override picture count for quick tests; default is full file.
    tb_num_pics = os.environ.get("TB_NUM_PICS")
    requested_pics = int(tb_num_pics) if tb_num_pics else None
    stimuli = load_stimuli_frames(INPUT_FILE, requested_pics)
    run_pics = len(stimuli)

    print(
        "TB config: "
        f"INPUT_FILE={INPUT_FILE}, run_pics={run_pics}, "
        f"L0_STIM_MODE={L0_STIM_MODE}, L0_STIM_GAIN={L0_STIM_GAIN}, "
        f"INTER_PIC_IDLE_CYCLES={INTER_PIC_IDLE_CYCLES}, "
        f"RESET_BETWEEN_PICS={RESET_BETWEEN_PICS}, HARD_RESET_BETWEEN_PICS={HARD_RESET_BETWEEN_PICS}, "
        f"DISABLE_RRAM_DEGRADATION={DISABLE_RRAM_DEGRADATION}, "
        f"RRAM_AGE_S={RRAM_AGE_S}, RRAM_LRS_KEEP_THRESHOLD={RRAM_LRS_KEEP_THRESHOLD}"
    )
    if run_pics <= 1:
        print("WARNING: run_pics <= 1, prediction confidence will be limited. Prefer TB_NUM_PICS>=5.")
    summarize_stimuli_activity(stimuli)
    
    # Initialize spike output matrices for each layer
    spike_out_layer_0 = [[0 for _ in range(NUM_CORES_LAYER_0 * NUM_NEURON)] for __ in range(run_pics)]
    spike_out_layer_1 = [[0 for _ in range(NUM_CORES_LAYER_1 * NUM_NEURON)] for __ in range(run_pics)]
    spike_out_layer_2 = [[0 for _ in range(NUM_CORES_LAYER_2 * NUM_NEURON)] for __ in range(run_pics)]
    
    # --- Clock and Reset Initialization ---
    print("\nStarting Clock and Reset\n")
    clock = Clock(dut.wb_clk_i, PERIOD, unit="ns")
    cocotb.start_soon(clock.start(start_high=True))
    
    # Initial state (assert reset)
    dut.wb_rst_i.value = 1
    dut.wbs_cyc_i.value = 0
    dut.wbs_stb_i.value = 0
    dut.wbs_we_i.value = 0
    dut.wbs_sel_i.value = 0b0000
    dut.wbs_adr_i.value = 0
    dut.wbs_dat_i.value = 0
    await RisingEdge(dut.wb_clk_i)
    
    # De-assert reset after a short delay
    await Timer(PERIOD * 1, unit="ns")
    dut.wb_rst_i.value = 0
    
    # --- Layer 0 Simulation ---
    # The cores in Layer 0 are indexed 0 to NUM_CORES_LAYER_0 - 1, corresponding to connection files 0 to 12.
    print("\n########################## START LAYER 0 ########################")
    layer = 0
    for core_layer_0 in range(NUM_CORES_LAYER_0):
        # 1. Configuration (ONCE)
        connection_layer_0 = connection_matrices[core_layer_0]
        await program_layer_connections(dut, core_layer_0, connection_layer_0, NUM_NEURON_LAYER_0)

        # 2. Simulation (EVERY PIC)
        # L0 input: stimuli (stimuli matrix, no core-specific indexing needed)
        await run_layer_for_all_pics(dut, core_layer_0, layer, NUM_CORES_LAYER_0, None, spike_out_layer_0, run_pics, stimuli=stimuli)

    print("\n########################## FINISH LAYER 0 ########################")
    print(f"L0 output: {spike_out_layer_0}")
    summarize_layer_activity("L0", spike_out_layer_0)
    await Timer(PERIOD * 1, unit="ns")

    # --- Layer 1 Simulation ---
    # The cores in Layer 1 are indexed 0 to NUM_CORES_LAYER_1 - 1, corresponding to connection files 13 to 16.
    print("\n########################## START LAYER 1 ########################")
    layer = 1
    for core_layer_1 in range(NUM_CORES_LAYER_1):
        # The connection index starts from 13
        conn_idx = 13 + core_layer_1
        connection_layer_1 = connection_matrices[conn_idx]
        
        # 1. Configuration (ONCE)
        await program_layer_connections(dut, core_layer_1, connection_layer_1, NUM_NEURON_LAYER_1)
        
        # 2. Simulation (EVERY PIC)
        # L1 input: spike_out_layer_0
        await run_layer_for_all_pics(dut, core_layer_1, layer, NUM_CORES_LAYER_1, spike_out_layer_0, spike_out_layer_1, run_pics, layer_axon_limit=NUM_AXON_LAYER_1)

    print("\n########################## FINISH LAYER 1 ########################")
    print(f"L1 output: {spike_out_layer_1}")
    summarize_layer_activity("L1", spike_out_layer_1)
    await Timer(PERIOD * 1, unit="ns")

    # --- Layer 2 Simulation ---
    # The cores in Layer 2 are indexed 0 to NUM_CORES_LAYER_2 - 1, corresponding to connection files 26_part1 to 26_part4.
    print("\n########################## START LAYER 2 ########################")
    layer = 2
    for core_layer_2 in range(NUM_CORES_LAYER_2):
        # The connection index is a float (e.g., 26.1, 26.2) for the parts
        conn_idx = 26 + (core_layer_2 + 1) / 10
        connection_layer_2 = connection_matrices[conn_idx]
        
        # 1. Configuration (ONCE)
        await program_layer_connections(dut, core_layer_2, connection_layer_2, NUM_NEURON_LAYER_2)

        # 2. Simulation (EVERY PIC)
        # L2 input: spike_out_layer_1. Note: core_idx is not used for L2 input indexing in run_layer_for_all_pics
        # because the original code suggests the L1 output is consolidated and indexed linearly for L2 input.
        await run_layer_for_all_pics(dut, core_layer_2, layer, NUM_CORES_LAYER_2, spike_out_layer_1, spike_out_layer_2, run_pics, layer_axon_limit=NUM_AXON_LAYER_2)

    print("\n########################## FINISH LAYER 2 ########################")
    print(f"L2 output: {spike_out_layer_2}")
    summarize_layer_activity("L2", spike_out_layer_2)
    await Timer(PERIOD * 1, unit="ns")

    # --- Final Results Calculation ---
    correct_pic = 0
    predict_class = calculate_majority_class(spike_out_layer_2)
    if predict_class and predict_class[0] >= 0:
        print(f"\nPrediction: Gesture Class {predict_class[0]}")
    else:
        print("\nPrediction: UNKNOWN (low confidence or saturation detected)")
        
    print("\nTest Completed.")