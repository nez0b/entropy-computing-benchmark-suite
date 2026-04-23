"""
Quick examples
--------------
python calculate.py data.csv                 # all defaults
python calculate.py data.csv -R 50 -p 4      # override R and num_pulses
python calculate.py -h                       # show full help
"""

import socket
import struct
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import argparse
from fractions import Fraction
from math import gcd
from time import sleep, time
from enum import Enum
from statistics import median

# Enum for Message Types
class MessageType(Enum):
    FREQUENCY_CALIBRATION = 0x0002
    DAC_CALIBRATION = 0x0001
    COMPUTATION = 0x0003
    COMPUTATION_TERMS = 0x0004
    STATUS = 0x0008
    
FPGA_IP = "10.0.1.2"
FPGA_PORT = 9090

# Response status codes
STATUS_SUCCESS = 0
STATUS_ERROR = -1
STATUS_TIMEOUT = -2

SEQUENCE_ID = 0  # Global sequence ID

HEADER_MESSAGE_TYPE_DAC = 0x0001
FRAME_SIZE_DAC = 0x14   # Frame size for DAC Calibration (20 bytes)

def get_args():
    parser = argparse.ArgumentParser(
        description="Process <filename> with configurable run‑time parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # required positional argument
    parser.add_argument(
        "filename",
        help="Path to the input file"
    )

    # optional arguments ────────────────────────────────────────────────
    parser.add_argument("-R", "--r", type=int, default=100,
                        help="R value")
    parser.add_argument("-s", "--num-samples", dest="num_samples",
                        type=int, default=1000,
                        help="Number of samples")
    parser.add_argument("-l", "--num-loops", dest="num_loops",
                        type=int, default=256,
                        help="Number of loops")
    parser.add_argument("-d", "--delay", type=int, default=86,
                        help="Delay value")

    parser.add_argument("-p", "--num-pulses", dest="num_pulses",
                        type=int, default=2,
                        help="Number of pulses")
    parser.add_argument("-w", "--pulse-width", dest="pulse_width",
                        type=int, default=1,
                        help="Width of each pulse")
    parser.add_argument("-b", "--distance-between-pulses",
                        dest="distance_between_pulses",
                        type=int, default=28,
                        help="Distance between consecutive pulses")
    parser.add_argument("-a", "--amplitude", type=int, default=585,
                        help="Pulse amplitude")
    parser.add_argument("-solps", "--solution-plot-show", dest="solution_plot_show",
                        type=int, default=1,
                        help="shows the plot with the solution")
    parser.add_argument("-ec", "--EDFA-current", dest="EDFA_current",
                        type=int, default=0,
                        help="captures EDFA current in mA")
    parser.add_argument("-tn", "--tnotes", dest="test_notes",
                        type=str, default="",
                        help="captures all the notes relevant for the test that are not considered as a variable e.g. , Vbias, etc")

    return parser.parse_args()
def create_frame(message_type, payload, verbose = False):
    """Creates a frame with an incremented sequence ID."""
    global SEQUENCE_ID
    frame_size = 8 + len(payload)
    header = struct.pack('>H I H', message_type.value, SEQUENCE_ID, frame_size)
    
    if verbose:
        print(f"Creating Frame: Type={message_type.name}, Sequence ID={SEQUENCE_ID}, Frame Size={frame_size}")
    
    SEQUENCE_ID += 1  
    return header + payload, SEQUENCE_ID - 1

def create_dac_calibration_frame_with_validity(fields, communication_version="1.1"):
    """Creates a DAC calibration frame with validity fields."""
    global SEQUENCE_ID

    num_pulses = fields.get('num_pulses', 0)# unsigned integer (16 bits / 2B)
    pulse_width = fields.get('pulse_width', 0) # unsigned integer (16 bits / 2B)
    distance_between_pulses = fields.get('distance_between_pulses', 0) # unsigned integer (16 bits / 2B)

    amplitude = fields.get('amplitude', 0) # signed integer (11 bits)- 7Vpp
    amplitude_valid = fields.get('amplitude_valid', False)
    if communication_version == "1.1":
        amplitude_field = ((0x8000 if amplitude_valid else 0x0000) | (amplitude & 0x07FF))
    else: # communication_version == "1.2"
        amplitude_field = (amplitude & 0x07FF)

    unused_amplitude = fields.get('unused_amplitude', 0) # signed integer (11 bits)- 7Vpp
    unused_amplitude_valid = fields.get('unused_amplitude_valid', False)
    if communication_version == "1.1":
        unused_amplitude_field = ((0x8000 if unused_amplitude_valid else 0x0000) | (unused_amplitude & 0x07FF))
    else: # communication_version == "1.2"
        unused_amplitude_field = (unused_amplitude & 0x07FF)

    processor_mode = fields.get('processor_mode', 0) # bool (1 bit)
    processor_mode_valid = fields.get('processor_mode_valid', False)
    if communication_version == "1.1":
        processor_mode_field = ((0x80 if processor_mode_valid else 0x00) | (processor_mode & 0x01))

    reset_dac = fields.get('reset_dac', 0) # bool (1 bit)
    reset_dac_valid = fields.get('reset_dac_valid', False)
    if communication_version == "1.1":
        reset_dac_field = ((0x80 if reset_dac_valid else 0x00) | (reset_dac & 0x01))

    validity_field = (
        (0x01 if processor_mode_valid else 0x00) |
        (0x02 if reset_dac_valid else 0x00) |
        (0x04 if amplitude_valid else 0x00) |
        (0x08 if unused_amplitude_valid else 0x00)
    )

    processor_reset_field = (
        (processor_mode & 0x01) |
        ((reset_dac & 0x01) << 1)
    )

    if communication_version == "1.1":
        frame = struct.pack('>H I H H H H H H B B',
                            HEADER_MESSAGE_TYPE_DAC, 
                            SEQUENCE_ID,
                            FRAME_SIZE_DAC,
                            num_pulses, 
                            pulse_width, 
                            distance_between_pulses, 
                            amplitude_field, 
                            unused_amplitude_field, 
                            processor_mode_field, 
                            reset_dac_field)
    else: # communication_version == "1.2"
        frame = struct.pack('>H I H B B H H H H H',
                    HEADER_MESSAGE_TYPE_DAC,  # 0-1
                    SEQUENCE_ID,              # 2-5
                    FRAME_SIZE_DAC,           # 6-7
                    validity_field,           # 8
                    processor_reset_field,    # 9
                    num_pulses,               # 10-11
                    pulse_width,              # 12-13
                    distance_between_pulses,  # 14-15
                    amplitude_field,          # 16-17
                    unused_amplitude_field)   # 18-19
    
    SEQUENCE_ID += 1  
    return frame, SEQUENCE_ID - 1

def create_computation_frame(num_variables, num_loops, response_delay, sum_constraint,
                         adc_fine_delay=0, integer_mode=0, noise_subtract=False, emulate=False, communication_version="1.1"):
    if communication_version == "1.1":
        valid_bits = 0x0F  # All fields valid
    else:
        valid_bits = 0xFF  # All fields valid
    start_computation = 0x01  # Start computation
    noise_emulate_field = (
        (0x01 if noise_subtract else 0x00) |
        (0x02 if emulate else 0x00)
    )
    if communication_version == "1.1":
        payload = struct.pack('>BBHHH', valid_bits, start_computation, num_variables, num_loops, response_delay)
        payload += struct.pack('>I', sum_constraint)
        print(f"Computation Frame Payload: valid_bits={valid_bits}, start_computation={start_computation}, num_variables={num_variables}, num_loops={num_loops}, response_delay={response_delay}, sum_constraint={sum_constraint}")
    else: # communication_version == "1.2"
        payload = struct.pack('>B B H H H I', valid_bits, start_computation, num_variables, num_loops, response_delay, sum_constraint)
        payload += struct.pack('>B B B', adc_fine_delay, integer_mode, noise_emulate_field)
        print(f"Computation Frame Payload: valid_bits={valid_bits}, start_computation={start_computation}, num_variables={num_variables}, num_loops={num_loops}, response_delay={response_delay}, sum_constraint={sum_constraint}, adc_fine_delay={adc_fine_delay}, integer_mode={integer_mode}, noise_subtract={noise_subtract}, emulate={emulate}")
    return create_frame(MessageType.COMPUTATION, payload)

def create_computation_terms_frame(num_variables, order, quadratic_index, values):
    reserved = 0x0000
    payload = struct.pack('>HHHH', num_variables, order, quadratic_index, reserved)
    payload += struct.pack('>' + 'i' * len(values), *values)  # Pack integer values
    
    
    if bool(args.solution_plot_show) == True:
        print(f"Computation Terms Frame Payload: num_variables={num_variables}, order={order}, quadratic_index={quadratic_index}, reserved={reserved}, values={values}")
    return create_frame(MessageType.COMPUTATION_TERMS, payload)

def create_status_frame(status_command_type):
    global SEQUENCE_ID

    payload = struct.pack('>H', status_command_type)
    print(f"Status Frame Payload: status_command_type=0x{status_command_type:04X}")

    return create_frame(MessageType.STATUS, payload)

def lcm(a, b):
    """Compute the least common multiple of a and b."""
    return a * b // gcd(a, b)

def compute_alpha(numbers):
    """
    Given a list of floats, compute the scaling factor alpha
    that when multiplied by each number gives an integer.
    """
    alpha = 1
    for x in numbers:
        # Convert float to a Fraction with a limited denominator
        frac = Fraction(x).limit_denominator()
        alpha = lcm(alpha, frac.denominator)
    return alpha

def read_csv_and_scale(filename, maximization = True):
    """
    Reads the CSV file.
      - First row corresponds to vector C.
      - Subsequent rows correspond to the matrix J.
    Returns:
      - C_int: list of integers for C.
      - J_int: matrix (list of lists) of integers for J.
      - alpha: the multiplier used.
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
   
    # Convert strings to floats
    C = [float(row[0]) for row in rows]
    J = [[float(x) for x in row[1:]] for row in rows]
   
    # Gather all values from C and J for alpha computation
    all_values = C + [val for row in J for val in row]
    alpha = compute_alpha(all_values)
   
    # Multiply each value by alpha and round to get integers and convert from min to max problem
    if maximization:
        C_int = [int(round(alpha * x)) for x in C]
        J_int = [[int(round(alpha * x)) for x in row] for row in J]
    else:
        C_int = [-int(round(alpha * x)) for x in C]
        J_int = [[-int(round(alpha * x)) for x in row] for row in J]
   
    return C_int, J_int, alpha


def parse_response_frame(response_frame, verbose = False):
    """Parses response header and validates sequence ID."""
    try:
        header = struct.unpack('>H I h', response_frame[:8])
        message_type, sequence_id, status = header

        if verbose:
            print(f"Response Header: {response_frame[:8].hex()}")
            print(f"Message Type: {MessageType(message_type).name}, Sequence ID: {sequence_id}, Status: {status}")

        if status != STATUS_SUCCESS:
            print(f"Error in response: Status Code {status}")
            return None  

        return sequence_id, message_type, response_frame[8:]

    except Exception as e:
        print(f"Failed to parse response frame: {e}")
        return None

def parse_computation_response(payload, R, num_variables, verbose = False, noise_compensation = True, noise = 10, qubo = False, communication_version = "1.1"):
    """Parses the computation response payload."""
    try:
        if communication_version == "1.1":
            fmt = f'>{num_variables}i'
            sigma_offset = 0
            sigma_values_ = struct.unpack(fmt, payload[:4*num_variables])  # 128 int32 values
        else: # communication_version == "1.2"
            print("I'm at 1.2")
            print(payload[0])
            print(payload[1])
            print(payload[2])
            print(payload[3])
            print(payload[4])
            print(payload[5])
            print(payload[6])
            print(payload[7])
            print(payload[8])
            print(payload[9])
            print(payload[10])
            print(payload[11])
            print(payload[12])
            num_variables_unpacked = struct.unpack('>H', payload[0:2])[0]  # Read number of variables from the first 2 bytes
            print("num_variables: ", num_variables_unpacked)
            compute_type_reserved = struct.unpack('> H H H', payload[2:8])  # Read the next 6 bytes for H, H, H
            print(compute_type_reserved)
            sigma_offset = 8 
            fmt = f'>{num_variables_unpacked}i'
            sigma_values_ = struct.unpack(fmt, payload[sigma_offset:sigma_offset + 4*num_variables])  # 128 int32 values
        if noise_compensation:
            sigma_values = []
            for sigma in sigma_values_:
                if sigma-median(sigma_values_) < 0:
                    sigma_values.append(0)
                else:
                    sigma_values.append(sigma-median(sigma_values_))
        else:
            sigma_values = sigma_values_
        if verbose:
           print("Computation Response - Sigma Values:")
           for i, sigma in enumerate(sigma_values):
               print(f"  σ{i}: {sigma}")
        sigma_sum = sum(sigma_values)
        sigma_normalized = []
        for i, sigma in enumerate(sigma_values):
            if R != 0:
                if sigma_sum:
                    if verbose:
                        print("  σ'",i,":", "{:.2f}".format((sigma*1.*R)/sigma_sum))
                    if not qubo:
                        sigma_normalized.append((sigma*1.*R)/sigma_sum)
                    else:
                        if i % 2 == 0:
                            sigma_normalized.append((sigma*1.*R)/(sigma_values[i]+sigma_values[i+1]))
                        else:
                            sigma_normalized.append((sigma*1.*R)/(sigma_values[i-1]+sigma_values[i]))
                else:
                    print("Sum of sigma is zero")
                    sigma_normalized.append(0)
            else:
                if verbose:
                    print("  σ'",i,":", "{:.2f}".format(sigma*1.))
                sigma_normalized.append(sigma*1.)
        return sigma_normalized
    except Exception as e:
        print(f"Failed to parse Computation response payload: {e}")

def parse_status_response(payload, verbose=False):
    try:
        if len(payload) < 4:
            raise ValueError("Payload too short to contain status response fields.")

        status_type, status_value = struct.unpack('>Hh', payload[0:4])  # uint16 + int16

        status_names = {
            0x0001: "Protocol Version: MAJOR",
            0x0002: "Protocol Version: MINOR",
            0x0003: "Protocol Version: PATCH",
            0x0004: "FPGA Build Number",
        }

        status_label = status_names.get(status_type, f"Reserved/Unknown Type (0x{status_type:04X})")

        if verbose:
            print(f"Status Response Type: {status_label}")
            print(f"Status Value: {status_value}")

        return status_type, status_value, status_label

    except Exception as e:
        print(f"Failed to parse Status response payload: {e}")
        return None


def send_command_and_wait_for_response(udp_socket, command_frame, expected_sequence_id, description, R=0, num_variables = 100, verbose = False, communication_version = "1.1") -> list or None:
    """Sends a command and waits for a response with the correct sequence ID."""
    try:
        start=time()
        udp_socket.sendto(command_frame, (FPGA_IP, FPGA_PORT))
        if verbose:
            print(f"Sent {description} Frame (Hex): {command_frame.hex()}")

        while True:
            try:
                response_frame, sender_address = udp_socket.recvfrom(1024)
                end = time()
                if verbose:
                    print("Compute time:",(end-start)*1e6,"us")                
                    print(f"Received response from {sender_address}: {response_frame.hex()}")

                parsed_response = parse_response_frame(response_frame, verbose = verbose)
                if parsed_response:
                    response_sequence_id, message_type, payload = parsed_response
                    if response_sequence_id == expected_sequence_id:
                        if message_type == MessageType.COMPUTATION.value:
                            sigma_normalized = parse_computation_response(payload, R, num_variables, verbose=True, communication_version = communication_version)
                            return sigma_normalized
                        elif message_type == MessageType.STATUS.value:
                            status_type, status_value, status_label = parse_status_response(payload, verbose=False)
                            return status_value

                        else:
                            return 
                    else:
                        print(f"Out-of-order response (Seq ID {response_sequence_id}), expected {expected_sequence_id}. Ignoring.")
                break
            except socket.timeout:
                print(f"No response for {description}, retrying...")
                #udp_socket.sendto(command_frame, (FPGA_IP, FPGA_PORT))

    except Exception as e:
        print(f"Error: {e}")

def calculate_energy (sigma, C, J):
    """
    Calculate the energy of the system given sigma, C, and J.
    H = sigma^T J sigma + C^T sigma
    """
    try:
        linear_term = np.dot(C, sigma)
        quadratic_term = np.dot(sigma, np.dot(J, sigma))
    except TypeError:
        linear_term = 0
        quadratic_term = 0
    return linear_term + quadratic_term

if __name__ == "__main__":
    communication_version = "1.1"
    # Example input
    args = get_args()
    # Demo: print parsed values (replace with your real logic)
    print(f"filename:               {args.filename}")
    print(f"R:                      {args.r}")
    print(f"num_samples:            {args.num_samples}")
    print(f"num_loops:              {args.num_loops}")
    print(f"delay:                  {args.delay}")
    print(f"num_pulses:             {args.num_pulses}")
    print(f"pulse_width:            {args.pulse_width}")
    print(f"distance_between_pulses:{args.distance_between_pulses}")
    print(f"amplitude:              {args.amplitude}")
    filename = args.filename
    R = args.r
    num_samples = args.num_samples
    num_loops = args.num_loops
    delay = args.delay
    dac_fields = {  
            'num_pulses': args.num_pulses,
            'pulse_width': args.pulse_width,
            'distance_between_pulses': args.distance_between_pulses,
            'amplitude': args.amplitude,
            'amplitude_valid': True,
            'unused_amplitude': 0,
            'unused_amplitude_valid': False,
            'processor_mode': 1,
            'processor_mode_valid': True,
            'reset_dac': 0,
            'reset_dac_valid': False
        }

    # filename = os.path.expanduser("~/Downloads/sensitivitymatrix50_padwithones.csv")
    # filename = os.path.expanduser("~/Downloads/stepmatrix.csv")
    # filename = os.path.expanduser("~/Downloads/sensitivitymatrix50_nondefaultzero.csv")
    # filename = os.path.expanduser("~/Downloads/Maximize_J_N30.csv")
    # filename = os.path.expanduser("~/Downloads/Maximize_J_N100_ratio_of_ones0.2_onesindex_50_97.csv")
    # filename = os.path.expanduser("~/Downloads/QCP_N_variables50_R_constrain100_rngseed42.csv")

    C_int, J_int, alpha = read_csv_and_scale(filename)
    # C_int = [1, 3, 6] # Sending all values in one frame
    # J_int = [[1, 0, 0], [0, 2, 0], [0, 0, 3]] # Sending each row in one frame
    # alpha=1
    P=1000
    #J_int = [[3,-P,-1,0,-1,0,-1,0],
    #          [-P,0,0,0,0,0,0,0],
    #          [-1,0,3,-P,-1,0,-1,0],
    #          [0,0,-P,0,0,0,0,0],
    #          [-1,0,-1,0,3,-P,-1,0],
    #          [0,0,0,0,-P,0,0,0],
    #          [-1,0,-1,0,-1,0,3,-P],
    #          [0,0,0,0,0,0,-P,0]]
    #J_int = [[-3,-P,1,0,1,0,1,0],
    #         [-P,0,0,0,0,0,0,0],
    #         [1,0,-3,-P,1,0,1,0],
    #         [0,0,-P,0,0,0,0,0],
    #         [1,0,1,0,-3,-P,1,0],
    #         [0,0,0,0,-P,0,0,0],
    #         [1,0,1,0,1,0,-3,-P],
    #         [0,0,0,0,0,0,-P,0]]
    #print(J_int)
    #C_int = [0,0,0,0,0,0,0,0]
    #alpha = 1

    num_variables = len(C_int)
    
    if bool(args.solution_plot_show) == True:
        print("C =", C_int)
        print("J =", J_int)    
        print("alpha:", alpha)
 
    # Open UDP socket once
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.settimeout(2)

        # Get Major, Minor, Patch version, build number
        frame, seq_id = create_status_frame(0x0001)  # Protocol Version: MAJOR
        major_version = send_command_and_wait_for_response(udp_socket, frame, seq_id, "Status Major Version")
        frame, seq_id = create_status_frame(0x0002)  # Protocol Version: MINOR
        minor_version = send_command_and_wait_for_response(udp_socket, frame, seq_id, "Status Minor Version")
        frame, seq_id = create_status_frame(0x0003)  # Protocol Version: PATCH
        patch_version = send_command_and_wait_for_response(udp_socket, frame, seq_id, "Status Patch Version")
        frame, seq_id = create_status_frame(0x0004)  # FPGA Build Number
        build_number = send_command_and_wait_for_response(udp_socket, frame, seq_id, "Status FPGA Build Number")
        print(f"FPGA Protocol Version: {major_version}.{minor_version}.{patch_version}, Build Number: {build_number}")
        communication_version = f"{major_version}.{minor_version}"
        full_version = f"{major_version}.{minor_version}.{patch_version}"

        
        # Send DAC calibration frame
        frame, seq_id = create_dac_calibration_frame_with_validity(dac_fields, communication_version=communication_version)
        send_command_and_wait_for_response(udp_socket, frame, seq_id, "DAC Calibration", communication_version=communication_version)
        
        # Computation Terms for C (Linear)
        frame, seq_id = create_computation_terms_frame(num_variables, 1, 0, C_int)
        send_command_and_wait_for_response(udp_socket, frame, seq_id, "Computation Terms (Linear)", communication_version=communication_version)
        
        # Send computation terms for J, one frame per row
        for i in range(num_variables):
            frame, seq_id = create_computation_terms_frame(num_variables, 2, i, J_int[i])
            send_command_and_wait_for_response(udp_socket, frame, seq_id, f"Computation Terms (Quadratic, Row {i})", communication_version=communication_version)
            
        # Send computation frame
        frame, seq_id = create_computation_frame(num_variables, num_loops, delay, R, communication_version=communication_version)
        sigma_normalized_list = []
        energy_list = []
        start_time = time()
        for a in range(num_samples):
            sigma_normalized = send_command_and_wait_for_response(udp_socket, frame, seq_id, "Computation Start", R, num_variables, verbose = False, communication_version=communication_version)
            print(sigma_normalized)
            sigma_normalized_list.append(sigma_normalized)
            energy_list.append(calculate_energy(sigma_normalized_list[a], C_int, J_int))
        computation_time = time() - start_time
        print("Computation time", computation_time*1000, "ms")

sigma_arr = np.array(sigma_normalized_list)

# Find index of highest energy solution
index = np.argmax(energy_list)

# Compute the average (mean) and standard deviation across samples (axis=0)
averages = np.mean(sigma_arr, axis=0)
std_devs = np.std(sigma_arr, axis=0)
best = sigma_arr[index, :]


if bool(args.solution_plot_show) == True:
    print("Best solution:")
    print(sigma_normalized_list[index])
    print("Average:")
    print(averages)
    print("Average energy:")
    print(np.mean(energy_list))
    print("Standard deviations:")
    print(std_devs)

print("Best energy:")
print(energy_list[index])
print("Delay")
print(args.delay)

# Create an x-axis for the indices
indices = np.arange(len(averages))

# Extract the short filename (remove directory and '.csv' extension)
short_filename = os.path.splitext(os.path.basename(filename))[0]

# -------------------------
# Setup date and filename info
# -------------------------
now = datetime.datetime.now()
date_time_str = now.strftime("%Y%m%d_%H%M%S")  # e.g. "20250320_142530"
save_filename = f"plot__{short_filename}__{date_time_str}.png"     # no spaces in the filename
save_filename2 = f"solution__{short_filename}__{date_time_str}.png"     # no spaces in the filename



# Prepare the information text for the info box
info_text = (
    f"Problem parameters\n"
    f"Filename: {short_filename}\n"
    f"Date and time: {date_time_str}\n"
    f"Sum Constrain (R): {R}\n"
    f"Number of variables: {num_variables}\n"
    f"Number of samples: {num_samples}\n"
    f"Number of loops: {num_loops}\n"
    f"Version: {full_version}\n"
    f"Build number: {build_number}\n"
    f"\n"

    f"Cavity parameters\n"
    f"Delay: {delay}\n"
    f"Pulsewidth:{args.pulse_width}\n"
    f"distance_between_pulses:{args.distance_between_pulses}\n"
    f"Maximum Amplitude:{args.amplitude}\n"
    f"EDFA current:{args.EDFA_current}\n"    
    f"Notes:{args.test_notes}\n"    
)

# Prepare the information text for the problem_info box
sol_info_text = (
    f"Solution:\n"
    f"max value in C: {sigma_arr[index, :].max()}\n"
    f"location in C: {sigma_arr[index, :].argmax()}\n"
    f"Maximum energy: {max(energy_list)}\n"
    f"Compute time: {computation_time*1000:.2f} ms\n"   
)

# Prepare text for Linear term and Quadratic term (each element/row on its own line)
linear_str = "Linear term:\n" + " ".join(f"{x:>6}" for x in C_int)
best_round = []
for i in best:
    best_round.append(int(round(i)))
solution_str = "Best solution:\n" + " ".join(f"{x:>6}" for x in best_round)
linear_str =  linear_str + "\n\n" + solution_str
quadratic_str = "Quadratic term:\n" + "\n".join(" ".join(f"{x:>6}" for x in row) for row in J_int)

# --------------------------------------------
# Create a figure with left plots and a nested right column for text boxes
# --------------------------------------------
fig = plt.figure(figsize=(12, 12))
# Create a gridspec with 2 rows and 2 columns:
# - Left column (width ratio 3) will hold the plots (2 rows)
# - Right column (width ratio 1) will be subdivided into 3 rows for text boxes
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])

# Left column: First subplot (top left): Bar chart with error bars
ax1 = fig.add_subplot(gs[0, 0])
indices = np.arange(len(averages))
ax1.bar(indices, averages, yerr=std_devs, capsize=5, align='center', alpha=0.7)
ax1.scatter(indices, sigma_normalized_list[index], color='red', label='Best Solution')
ax1.text(sigma_arr[index, :].argmax() + 0.1, sigma_arr[index, :].max() + 0.1, f"({sigma_arr[index, :].argmax()}, {sigma_arr[index, :].max():0.2f})")
ax1.set_xlabel("Element Index")
ax1.set_ylabel("Average Value")
ax1.grid()
ax1.set_title("Average Values with Error Bars")
ax1.set_xticks(indices)
ax1.legend()

# Left column: Second subplot (bottom left): Histogram of the energy list
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(energy_list, bins=20, alpha=0.7)
ax2.set_xlabel("Energy")
ax2.set_ylabel("Frequency")
ax2.set_title("Histogram of Energy List")

# Right column: Create a nested gridspec with 3 rows for text boxes
gs_right = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 1])

# Top text box: Info box with simulation results
ax_info = fig.add_subplot(gs_right[0])
ax_info.axis('off')
ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# Top text box: Info box with simulation results
ax_info = fig.add_subplot(gs_right[1])
ax_info.axis('off')
ax_info.text(0.05, 0.95, sol_info_text, transform=ax_info.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))



# Middle text box: Display Linear term (C_int) fully line by line
ax_C = fig.add_subplot(gs_right[2])
ax_C.axis('off')
ax_C.text(0.05, 0.95, linear_str, transform=ax_C.transAxes, fontsize=100/num_variables,
          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Bottom text box: Display Quadratic term (J_int) fully line by line
ax_J = fig.add_subplot(gs_right[3])
ax_J.axis('off')
ax_J.text(0.05, 0.95, quadratic_str, transform=ax_J.transAxes, fontsize=100/num_variables,
          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()

# --------------------------------------------
# Save results to a file
# --------------------------------------------
results_filename = f"results__{short_filename}__{date_time_str}.npz"

np.savez(results_filename,
         sigma_array=sigma_arr,
         best_solution=sigma_normalized_list[index],
         best_energy=energy_list[index],
         best_index=index,
         energy_list=energy_list,
         averages=averages,
         std_devs=std_devs,
         R=R,
         num_samples=num_samples,
         num_loops=num_loops,
         delay=delay,
         num_variables=num_variables,
         filename=filename,
         EDFA_current=args.EDFA_current,
         alpha=alpha,
         linear_term=C_int,
         quadratic_term=J_int,
         computation_time=computation_time)

print("Results saved as:", results_filename)

# Load with
# data = np.load("results_20250425_143012.npz", allow_pickle=True)
# print(data["best_solution"])


# -------------------
# Save the figure and display it
# -------------------
plt.savefig(save_filename)
print("Plot saved as:", save_filename)
plt.show(block=bool(args.solution_plot_show))
print ("Displaying plot results:",bool(args.solution_plot_show))
# if bool(args.solution_plot_show) == False: 
plt.close() 
