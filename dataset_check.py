"""Quick dataset sanity check."""
from __future__ import annotations
from dataset import ContactEventDataset

ds = ContactEventDataset("data/stick_slip", task="stick_slip", seq_len=128)
s = ds[0]
print("vision", tuple(s["vision"].shape))
print("force", tuple(s["force"].shape))
print("velocity", tuple(s["velocity"].shape))
print("proprio", tuple(s["proprio"].shape))
print("label", tuple(s["label"].shape))
print("OK")
