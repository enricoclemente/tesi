import commands
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TensorBoardPlotter():
    def __init__(self, save_folder="./logs/exp"):
        self.writer = SummaryWriter(save_folder)

def strip_empty_str(strings):
    while strings and strings[-1] == "":
        del strings[-1]
    return strings

def fetch_gpu_status():
    """ Run nvidia-smi and parse the output
    requires Python 2 only dependency and can be launched on unix only!
    Taken from github: https://gist.github.com/rueberger/f903e9261454cc0c6cdf6243dbf2b0e5
    """

    status_code, output = commands.getstatusoutput('nvidia-smi')
    assert status_code == 0

    gpu_records = []
    gpu_line = False
    process_block = False
    valid_ids = []
    for line in output.split('\n'):
        # if the line looks like this (possibly with a different GPU name)
        # |   1  TITAN X (Pascal)    Off  | 0000:03:00.0     Off |                  N/A |
        # these two cases cover all of the workstations and the slowpokes currently (3/17)
        if 'TITAN X' in line or 'GeForce' in line:
            gpu_line = True
            gpu_id = int(strip_empty_str(line.split('|')[1].split(' '))[0])
            continue
        # if the line looks like this
        # | 79%   87C    P2   221W / 250W |   2291MiB / 12189MiB |    100%      Default |
        if gpu_line:
            _, physicals, memory, usage, _ = line.split('|')

            physicals_tokens = strip_empty_str(physicals.split(' '))
            memory_tokens = strip_empty_str(memory.split(' '))
            usage_tokens = strip_empty_str(usage.split(' '))

            record = {
                'physicals': {
                    'fan': physicals_tokens[0],
                    'temp': physicals_tokens[1],
                    'power': physicals_tokens[3]
                },
                'memory_frac': float(memory_tokens[0][:-3]) / float(memory_tokens[2][:-3]),
                'tot_memory': int(memory_tokens[2][:-3]),
                'usage_frac': float(usage_tokens[0][:-1]) / 100.,
                'id': gpu_id,
                'processes': []
            }
            gpu_records.append(record)
            valid_ids.append(gpu_id)
            gpu_line = False
            continue

        # if the line looks like this
        # | Processes:                                                       GPU Memory |
        if 'Processes:' in line:
            process_block = True
            indexed_gpus = {rec['id']: rec for rec in gpu_records}
            continue

        # if the line looks like this
        # |    1     89398    C   python                                        2289MiB |-
        if process_block:
            disallowed_lines = ['+----', '|  GPU', '|======', '|    0 ', '|  No running processes found']
            invalid_line = np.asarray([line.startswith(illegal_line) for illegal_line in disallowed_lines]).any()
            if invalid_line:
                continue
            gpu_id, pid, ptype, name, mem_usage = strip_empty_str(strip_empty_str(line.split('|'))[0].split(' '))
            gpu_id = int(gpu_id)
            pid = int(pid)
            if gpu_id in valid_ids:
                process_record = {
                    'pid': pid,
                    'type': ptype,
                    'name': name,
                    'mem_frac': float(mem_usage[:-3])  / indexed_gpus[gpu_id]['tot_memory']
                }
                indexed_gpus[gpu_id]['processes'].append(process_record)
            continue
    return gpu_records