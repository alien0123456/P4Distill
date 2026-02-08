import json
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader


class FlowDataset(Dataset):

    def __init__(self, args, filename: str):
        super().__init__()
        self.args = args
        self.flows: List[Dict[str, Any]] = []

        with open(filename, "r", encoding="utf-8", errors="ignore") as fp:
            instances = json.load(fp)

        for ins in instances:
            len_seq = ins["len_seq"]
            ts_seq = ins["ts_seq"]

            len_seq = [min(int(x), args.pkt_len_vocab_size - 1) for x in len_seq]

            ipd_seq = [0.0]
            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])

            ipd_seq_q = []
            for x in ipd_seq:
                v = round(float(x) * 10000.0)  # unit: 0.1ms (1e-4 s) => 100us
                if v < 0:
                    raise AssertionError(f"ipd_seq quantized < 0: {v}, raw={x}")
                v = min(int(v), args.ipd_vocab_size - 1)
                ipd_seq_q.append(v)

            if len(len_seq) > 4096:
                len_seq = len_seq[:4096]
                ipd_seq_q = ipd_seq_q[:4096]

            self.flows.append({
                "label": int(ins["label"]),
                "len_seq": len_seq,
                "ipd_seq": ipd_seq_q,
            })

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index: int):
        flow = self.flows[index]
        return flow["len_seq"], flow["ipd_seq"], flow["label"]


def make_collate_fn(args):

    w = int(args.seq_window_size)

    def collate_fn(batch: List[Tuple[List[int], List[int], int]]):
        len_x_batch: List[List[int]] = []
        ipd_x_batch: List[List[int]] = []
        label_batch: List[int] = []

        for len_seq, ipd_seq, label in batch:
            flow_packets = len(len_seq)
            if flow_packets < w:
                raise Exception("Flow packets < window size!!!")


            for idx in range(0, flow_packets - w + 1):
                len_x_batch.append(len_seq[idx: idx + w])
                ipd_x_batch.append(ipd_seq[idx: idx + w])
                label_batch.append(int(label))

        if len(label_batch) == 0:
            len_t = torch.empty((0, w), dtype=torch.long)
            ipd_t = torch.empty((0, w), dtype=torch.long)
            lab_t = torch.empty((0,), dtype=torch.long)
        else:
            len_t = torch.tensor(len_x_batch, dtype=torch.long)
            ipd_t = torch.tensor(ipd_x_batch, dtype=torch.long)
            lab_t = torch.tensor(label_batch, dtype=torch.long)

        if getattr(args, "cuda_device_id", None) is not None:
            device = torch.device(f"cuda:{args.cuda_device_id}")
            len_t = len_t.to(device, non_blocking=True)
            ipd_t = ipd_t.to(device, non_blocking=True)
            lab_t = lab_t.to(device, non_blocking=True)

        return len_t, ipd_t, lab_t

    return collate_fn


def build_data_loader(
    args,
    filename: str,
    batch_size: int,
    is_train: bool = False,
    shuffle: bool = True,
    generator=None,
    worker_init_fn=None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
):
    dataset = FlowDataset(args, filename)
    collate_fn = make_collate_fn(args)

    if not is_train:
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    print(f"The size of {'train' if is_train else 'test'}_set is {len(dataset)}.")
    return loader

