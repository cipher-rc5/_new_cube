import argparse
import os

import torch
import trimesh

from cube3d.inference.engine import Engine, EngineFast
from cube3d.inference.utils import normalize_bbox, select_device
from cube3d.mesh_utils.postprocessing import (
    PYMESHLAB_AVAILABLE,
    create_pymeshset,
    postprocess_mesh,
    save_mesh,
)
from cube3d.renderer import renderer


def generate_mesh(
    engine,
    prompt,
    output_dir,
    output_name,
    resolution_base=8.0,
    disable_postprocess=False,
    top_p=None,
    bounding_box_xyz=None,
    chunk_size=250_000,
):
    mesh_v_f = engine.t2s(
        [prompt],
        use_kv_cache=True,
        resolution_base=resolution_base,
        chunk_size=chunk_size,
        top_p=top_p,
        bounding_box_xyz=bounding_box_xyz,
    )
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    obj_path = os.path.join(output_dir, f"{output_name}.obj")
    if PYMESHLAB_AVAILABLE:
        ms = create_pymeshset(vertices, faces)
        if not disable_postprocess:
            target_face_num = max(10000, int(faces.shape[0] * 0.1))
            print(f"Postprocessing mesh to {target_face_num} faces")
            postprocess_mesh(ms, target_face_num, obj_path)

        save_mesh(ms, obj_path)
    else:
        if not disable_postprocess:
            print(
                "pymeshlab not installed; exporting raw mesh via trimesh (postprocessing skipped). "
                "Install with `uv sync --extra meshlab` if a cp314 wheel is available on your platform."
            )
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(obj_path)

    return obj_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cube shape generation script")
    parser.add_argument(
        "--config-path",
        type=str,
        default="cube3d/configs/open_model_v0.5.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Path to the output directory to store .obj and .gif files",
    )
    parser.add_argument(
        "--gpt-ckpt-path",
        type=str,
        required=True,
        help="Path to the main GPT checkpoint file.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        required=True,
        help="Path to the shape encoder/decoder checkpoint file.",
    )
    parser.add_argument(
        "--fast-inference",
        help="Use optimized inference",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "mlx"],
        default="torch",
        help="Inference backend. 'mlx' is Apple Silicon only and currently covers the GPT decode loop.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generating a 3D mesh",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Float < 1: Keep smallest set of tokens with cumulative probability ≥ top_p. Default None: deterministic generation.",
    )
    parser.add_argument(
        "--bounding-box-xyz",
        nargs=3,
        type=float,
        help="Three float values for x, y, z bounding box",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--render-gif",
        help="Render a turntable gif of the mesh",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--disable-postprocessing",
        help="Disable postprocessing on the mesh. This will result in a mesh with more faces.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--resolution-base",
        type=float,
        default=8.0,
        help=(
            "Marching-cubes grid is 2**resolution_base cells per axis. "
            "Defaults to 8.0 (~257^3 ≈ 17M samples). Try 8.5 / 9.0 for higher "
            "fidelity; values above 9.0 are memory-bound on most GPUs."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help=(
            "Marching-cubes query batch size. Controls how many grid points are "
            "evaluated per launch during geometry extraction. Larger values use "
            "more MPS/GPU memory but reduce launch overhead. At resolution 8.5, "
            "1_000_000 OOMs MPS (>130 GiB); 250_000 is a safe ~2.5x speedup over "
            "the legacy 100_000. Lower further if you hit OOM at higher resolutions."
        ),
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = select_device()
    print(f"Using device: {device}")
    # Initialize engine based on the selected backend and fast_inference flag.
    if args.backend == "mlx":
        from cube3d.inference.mlx_engine import MlxEngine, MLX_AVAILABLE

        if not MLX_AVAILABLE or device.type != "mps":
            raise SystemExit("mlx backend requires Apple Silicon with mlx installed")
        engine = MlxEngine(
            args.config_path, args.gpt_ckpt_path, args.shape_ckpt_path
        )
    elif args.fast_inference and device.type == "cuda":
        print(
            "Using cuda graphs, this will take some time to warmup and capture the graph."
        )
        engine = EngineFast(
            args.config_path, args.gpt_ckpt_path, args.shape_ckpt_path, device=device
        )
        print("Compiled the graph.")
    else:
        if args.fast_inference:
            print("fast-inference is CUDA-only; falling back to standard engine")
        engine = Engine(
            args.config_path, args.gpt_ckpt_path, args.shape_ckpt_path, device=device
        )

    if args.bounding_box_xyz is not None:
        args.bounding_box_xyz = normalize_bbox(tuple(args.bounding_box_xyz))

    # Generate meshes based on input source
    obj_path = generate_mesh(
        engine,
        args.prompt,
        args.output_dir,
        "output",
        args.resolution_base,
        args.disable_postprocessing,
        args.top_p,
        args.bounding_box_xyz,
        args.chunk_size,
    )
    if args.render_gif:
        gif_path = renderer.render_turntable(obj_path, args.output_dir)
        print(f"Rendered turntable gif for {args.prompt} at `{gif_path}`")
    print(f"Generated mesh for {args.prompt} at `{obj_path}`")
