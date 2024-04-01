import platform
import torch
import numpy as np


def f(x):
	return x


def cuda_setup():
	os_name = platform.system()
	print("\n______________________________________________________")
	gpu_available = torch.cuda.is_available()
	print(f"Current OS: {os_name}.")
	print(f"Current cuda device: {torch.cuda.current_device()}.")
	print(f"NVIDIA GPU available: {gpu_available}.")
	print(f"Using {"GPU" if (gpu_available) else "CPU."}.")
	if (gpu_available):
		torch.set_default_device('cuda')

	print("______________________________________________________\n")


def main():
	cuda_setup()
	random_tensor_3x4 = torch.rand(3, 4)

	print(f"Shape of tensor: {random_tensor_3x4.shape}")
	print(f"Datatype of tensor: {random_tensor_3x4.dtype}")
	print(f"Device tensor is stored on: {random_tensor_3x4.device}")

	return 0


if __name__ == "__main__":
	main()
