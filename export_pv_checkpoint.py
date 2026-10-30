"""Export a best PV local checkpoint with the original distributed world size."""
import argparse
import os

import torch
import torch.distributed

from finetune import export_best_pv_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", required=True, help="Training output directory containing best_checkpoint/")
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")
    try:
        export_best_pv_checkpoint(args.save, device)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
