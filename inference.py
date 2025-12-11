# TO RUN INFERENCE ON CATVTON: python inference.py   --dataset_name vitonhd   --data_root_path VITON-HD --batch_size 16 --mixed_precision bf16 --output_dir output50-0.3 --cfg_rescale <CFG_RESCALING_FACTOR> --num_inference_steps 50 --scheduler [ddim / unipc / dpmsolver]


import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from PIL import Image, ImageFilter
import time

from model.pipeline import CatVTONPipeline
import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from PIL import Image, ImageFilter
import time
import psutil  # Add this import for CPU RAM monitoring
import threading  # For background monitoring

# Add this new class for memory monitoring
class MemoryMonitor:
    def __init__(self, interval=1.0):
        """
        Monitor CPU RAM and GPU memory usage.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
        # Storage for measurements
        self.cpu_ram_history = []
        self.gpu_mem_history = []
        self.timestamps = []
        
        # Peak values
        self.peak_cpu_ram = 0
        self.peak_gpu_mem = 0
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        start_time = time.time()
        while self.monitoring:
            # CPU RAM usage
            cpu_ram_mb = self.process.memory_info().rss / 1024 / 1024
            
            # GPU memory usage
            gpu_mem_mb = 0
            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Record
            self.cpu_ram_history.append(cpu_ram_mb)
            self.gpu_mem_history.append(gpu_mem_mb)
            self.timestamps.append(time.time() - start_time)
            
            # Update peaks
            self.peak_cpu_ram = max(self.peak_cpu_ram, cpu_ram_mb)
            self.peak_gpu_mem = max(self.peak_gpu_mem, gpu_mem_mb)
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Memory monitoring started...")
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Memory monitoring stopped.")
    
    def get_current_usage(self):
        """Get current memory usage"""
        cpu_ram = self.process.memory_info().rss / 1024 / 1024
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        return cpu_ram, gpu_mem
    
    def print_summary(self):
        """Print memory usage summary"""
        if not self.cpu_ram_history:
            print("No memory data collected.")
            return
        
        avg_cpu = np.mean(self.cpu_ram_history)
        avg_gpu = np.mean(self.gpu_mem_history)
        
        print("\n" + "="*80)
        print("MEMORY USAGE SUMMARY")
        print("="*80)
        print(f"CPU RAM:")
        print(f"  - Current: {self.cpu_ram_history[-1]:.2f} MB")
        print(f"  - Peak:    {self.peak_cpu_ram:.2f} MB")
        print(f"  - Average: {avg_cpu:.2f} MB")
        print(f"\nGPU Memory:")
        print(f"  - Current: {self.gpu_mem_history[-1]:.2f} MB")
        print(f"  - Peak:    {self.peak_gpu_mem:.2f} MB")
        print(f"  - Average: {avg_gpu:.2f} MB")
        
        if torch.cuda.is_available():
            total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            print(f"  - Total available: {total_gpu:.2f} MB")
            print(f"  - Peak utilization: {(self.peak_gpu_mem/total_gpu)*100:.1f}%")
        print("="*80)
    
    def save_history(self, filepath):
        """Save memory history to file"""
        import json
        data = {
            'timestamps': self.timestamps,
            'cpu_ram_mb': self.cpu_ram_history,
            'gpu_mem_mb': self.gpu_mem_history,
            'peak_cpu_ram_mb': self.peak_cpu_ram,
            'peak_gpu_mem_mb': self.peak_gpu_mem,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Memory history saved to {filepath}")


def print_memory_stats(stage=""):
    """Print current memory statistics"""
    process = psutil.Process()
    cpu_ram = process.memory_info().rss / 1024 / 1024
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        print(f"\n[{stage}] Memory Usage:")
        print(f"  CPU RAM: {cpu_ram:.2f} MB")
        print(f"  GPU Allocated: {gpu_allocated:.2f} MB")
        print(f"  GPU Reserved: {gpu_reserved:.2f} MB")
        print(f"  GPU Total: {gpu_total:.2f} MB")
        print(f"  GPU Utilization: {(gpu_allocated/gpu_total)*100:.1f}%")
    else:
        print(f"\n[{stage}] CPU RAM: {cpu_ram:.2f} MB")

class InferenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
    
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8) 
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True) 
        self.data = self.load_data()
    
    def load_data(self):
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, mask = [Image.open(data[key]) for key in ['person', 'cloth', 'mask']]
        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }

class VITONHDTestDataset(InferenceDataset):
    def load_data(self):
        assert os.path.exists(pair_txt:=os.path.join(self.args.data_root_path, 'test_pairs_unpaired.txt')), f"File {pair_txt} does not exist."
        with open(pair_txt, 'r') as f:
            lines = f.readlines()
        self.args.data_root_path = os.path.join(self.args.data_root_path, "test")
        output_dir = os.path.join(self.args.output_dir, "vitonhd", 'unpaired' if not self.args.eval_pair else 'paired')
        data = []
        for line in lines:
            person_img, cloth_img = line.strip().split(" ")
            if os.path.exists(os.path.join(output_dir, person_img)):
                continue
            if self.args.eval_pair:
                cloth_img = person_img
            data.append({
                'person_name': person_img,
                'person': os.path.join(self.args.data_root_path, 'image', person_img),
                'cloth': os.path.join(self.args.data_root_path, 'cloth', cloth_img),
                'mask': os.path.join(self.args.data_root_path, 'agnostic-mask', person_img.replace('.jpg', '_mask.png')),
            })
        return data

class DressCodeTestDataset(InferenceDataset):
    def load_data(self):
        data = []
        for sub_folder in ['upper_body', 'lower_body', 'dresses']:
            assert os.path.exists(os.path.join(self.args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."
            pair_txt = os.path.join(self.args.data_root_path, sub_folder, 'test_pairs_paired.txt' if self.args.eval_pair else 'test_pairs_unpaired.txt')
            assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."
            with open(pair_txt, 'r') as f:
                lines = f.readlines()

            output_dir = os.path.join(self.args.output_dir, f"dresscode-{self.args.height}", 
                                      'unpaired' if not self.args.eval_pair else 'paired', sub_folder)
            for line in lines:
                person_img, cloth_img = line.strip().split(" ")
                if os.path.exists(os.path.join(output_dir, person_img)):
                    continue
                data.append({
                    'person_name': os.path.join(sub_folder, person_img),
                    'person': os.path.join(self.args.data_root_path, sub_folder, 'images', person_img),
                    'cloth': os.path.join(self.args.data_root_path, sub_folder, 'images', cloth_img),
                    'mask': os.path.join(self.args.data_root_path, sub_folder, 'agnostic_masks', person_img.replace('.jpg', '.png'))
                })
        return data
                    
       
def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optimized CatVTON inference.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",
        help="The path to the base model.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help="The path to the checkpoint of trained tryon model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The datasets to use for evaluation.",
    )
    parser.add_argument(
        "--data_root_path", 
        type=str, 
        required=True,
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where predictions will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=555, help="A seed for reproducible evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="The batch size for evaluation."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to perform.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="The scale of classifier-free guidance for inference.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help="The resolution for input images"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="The resolution for input images"
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--eval_pair",
        action="store_true",
        help="Whether to evaluate the pair.",
    )
    parser.add_argument(
        "--concat_eval_results",
        action="store_true",
        help="Whether to concatenate all conditions into one image.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Whether to allow TF32 on Ampere GPUs for faster computation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,  # Reduced default to avoid worker warnings
        help="Number of subprocesses to use for data loading."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision."
    )
    
    # GPU-Compatible Optimization arguments
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Use torch.compile for JIT optimization (PyTorch 2.0+). Provides 10-30%% speedup."
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Compilation mode for torch.compile. 'max-autotune' is slowest to compile but fastest inference."
    )
    parser.add_argument(
        "--use_xformers",
        action="store_true",
        help="Use xFormers memory efficient attention (requires xformers installation)."
    )
    parser.add_argument(
        "--use_sdpa",
        action="store_true",
        default=True,
        help="Use Scaled Dot Product Attention (PyTorch 2.0+ native, recommended)."
    )
    parser.add_argument(
        "--use_channels_last",
        action="store_true",
        default=True,
        help="Use channels-last memory format for better GPU performance."
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Enable VAE tiling for processing very large images (reduces memory but slower)."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark to measure inference speed."
    )
    parser.add_argument(
        "--benchmark_samples",
        type=int,
        default=10,
        help="Number of samples to use for benchmarking."
    )

    parser.add_argument(
        "--concat_axis",
        type=str,
        choices=["x", "y", 'random'],
        default="y",
        help="The axis to concat the cloth feature.",
    )
    parser.add_argument(
        "--cfg_rescale",
        type=float,
        default=0.0,
        help="The CFG re-scaling factor.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "unipc", "dpmsolver"],
        help="The scheduler to use for inference.",
    )

    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and hasattr(args, 'local_rank') and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@torch.no_grad()
def main():
    args = parse_args()
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(interval=0.5)  # Monitor every 0.5 seconds
    memory_monitor.start()
    
    print("\n" + "="*80)
    print("GPU-OPTIMIZED CatVTON INFERENCE")
    print("="*80)
    print(f"Optimization settings:")
    print(f"  - Mixed precision: {args.mixed_precision}")
    print(f"  - Torch compile: {args.use_torch_compile}")
    if args.use_torch_compile:
        print(f"  - Compile mode: {args.compile_mode}")
    print(f"  - xFormers: {args.use_xformers}")
    print(f"  - SDPA (PyTorch 2.0+): {args.use_sdpa}")
    print(f"  - Channels-last format: {args.use_channels_last}")
    print(f"  - VAE tiling: {args.enable_vae_tiling}")
    print(f"  - TF32: {args.allow_tf32}")
    print(f"  - Batch size: {args.batch_size}")
    print("="*80 + "\n")
    
    print_memory_stats("Initial")
    
    # Pipeline with GPU-compatible optimizations
    pipeline = CatVTONPipeline(
        attn_ckpt_version=args.dataset_name,
        attn_ckpt=args.resume_path,
        base_ckpt=args.base_model_path,
        weight_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
        skip_safety_check=True,
        compile=args.use_torch_compile,
        use_tf32=args.allow_tf32,
        use_xformers=args.use_xformers,
        use_sdpa=args.use_sdpa,
        use_channels_last=args.use_channels_last,
        enable_vae_tiling=args.enable_vae_tiling,
        compile_mode=args.compile_mode,
        scheduler=args.scheduler
    )
    
    print_memory_stats("After Pipeline Init")
    
    # Dataset
    if args.dataset_name == "vitonhd":
        dataset = VITONHDTestDataset(args)
    elif args.dataset_name == "dresscode":
        dataset = DressCodeTestDataset(args)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset_name}.")
    
    print(f"Dataset {args.dataset_name} loaded, total {len(dataset)} pairs.\n")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    
    print_memory_stats("After Dataset Init")
    
    # Generator for reproducibility
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
        
    
    # Regular inference
    args.output_dir = os.path.join(
        args.output_dir, 
        f"{args.dataset_name}-{args.height}", 
        "paired" if args.eval_pair else "unpaired"
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Track timing
    total_start = time.time()
    num_processed = 0
    
    print("Starting inference...\n")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        person_images = batch['person']
        cloth_images = batch['cloth']
        masks = batch['mask']
        
        results = pipeline(
            person_images,
            cloth_images,
            masks,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_rescale=args.cfg_rescale
        )
        num_processed += len(results)
        
        # Print memory stats every 10 batches
        if (batch_idx + 1) % 10 == 0:
            cpu_ram, gpu_mem = memory_monitor.get_current_usage()
            print(f"\n[Batch {batch_idx+1}] CPU: {cpu_ram:.1f}MB, GPU: {gpu_mem:.1f}MB")
        
        if args.concat_eval_results or args.repaint:
            person_images = to_pil_image(person_images)
            cloth_images = to_pil_image(cloth_images)
            masks = to_pil_image(masks)
        
        for i, result in enumerate(results):
            person_name = batch['person_name'][i]
            output_path = os.path.join(args.output_dir, person_name)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            
            if args.repaint:
                person_path = dataset.data[batch['index'][i]]['person']
                mask_path = dataset.data[batch['index'][i]]['mask']
                person_image = Image.open(person_path).resize(result.size, Image.LANCZOS)
                mask = Image.open(mask_path).resize(result.size, Image.NEAREST)
                result = repaint(person_image, mask, result)
            
            if args.concat_eval_results:
                w, h = result.size
                concated_result = Image.new('RGB', (w*3, h))
                concated_result.paste(person_images[i], (0, 0))
                concated_result.paste(cloth_images[i], (w, 0))  
                concated_result.paste(result, (w*2, 0))
                result = concated_result
            
            result.save(output_path)
    
    total_time = time.time() - total_start
    
    # Stop monitoring
    memory_monitor.stop()
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Total images processed: {num_processed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/num_processed:.3f}s ({num_processed/total_time:.2f} fps)")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Print and save memory summary
    memory_monitor.print_summary()
    memory_monitor.save_history(os.path.join(args.output_dir, 'memory_history.json'))
    
    print_memory_stats("Final")


if __name__ == "__main__":
    main()