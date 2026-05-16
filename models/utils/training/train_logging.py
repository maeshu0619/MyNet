# models/utils/training/train_logging.py

import torch


def log_backend_summary(args, writer):
    compression_backend = str(
        getattr(args, "compression_loss_backend", "proxy")
    ).strip().lower()

    writer.write(f"Method Name: {getattr(args, 'method_name', 'Mine')}")
    writer.write(
        f"Surrogate Name: "
        f"{getattr(args, 'surrogate_name', getattr(args, 'compress', 'OctAttention'))}"
    )
    writer.write(f"Geometry Loss Type: {args.loss_type}")
    writer.write(f"Discrete Loss Mode: {args.discrete_loss_mode}")

    writer.write(
        "Optimization Modes: "
        f"geometry="
        f"{'ste_hard' if args.discrete_loss_mode == 'ste_hard' else ('weighted_soft' if args.discrete_loss_mode == 'weighted_soft' else 'hard')}, "
        f"compression={args.compression_loss_backend}"
    )

    writer.write(
        "Compression Delta Sign: "
        "delta_percent=(after_bits-before_bits)/before_bits*100; "
        "negative means improved compression."
    )

    writer.write(
        "Gradient Diagnostics: "
        f"compression_grad_probe={bool(getattr(args, 'compression_grad_probe', False))}"
        f"(every={int(getattr(args, 'compression_grad_probe_every', 1))}), "
        f"debug_grad_flow={bool(getattr(args, 'debug_grad_flow', False))}"
        f"(rate={int(getattr(args, 'debug_grad_flow_rate', 1))})"
    )

    writer.write(f"Compression Codec: {getattr(args, 'compress', 'OctAttention')}")
    writer.write(f"Compression Rate Metric: {args.compression_rate_metric}")
    writer.write(f"Compression Loss Backend: {args.compression_loss_backend}")

    if compression_backend.startswith("sparsepcgc"):
        writer.write("Compression Backend Detail: SparsePCGC mode")

    if compression_backend.startswith("gpcc"):
        writer.write("Compression Backend Detail: G-PCC mode")

    if compression_backend.startswith("draco"):
        writer.write("Compression Backend Detail: Draco mode")


def log_input_mode(args, writer):
    if bool(getattr(args, "train_patch_subset_enable", False)):
        writer.write("Model Input is Whole Point Cloud (Octree Subtree Mode)")
        writer.write(
            f"Train Patch Subset: enabled, "
            f"max_patches={getattr(args, 'train_patch_subset_max_patches', None)}"
        )
    elif args.split2patch:
        writer.write("Model Input is Patch")
        writer.write(
            f"Patch Batch Size: {getattr(args, 'patch_batch_size', None)}"
        )
    else:
        writer.write("Model Input is Whole Point Cloud")


def log_structure_debug(writer, structure_debug, step, num_steps):
    if not structure_debug:
        return

    writer.write(
        f"StructureStats step={step + 1}/{num_steps}: "
        f"repair_ratio={float(structure_debug.get('repair_ratio', 0.0)):.6f}, "
        f"add_ratio={float(structure_debug.get('add_ratio', 0.0)):.6f}, "
        f"add_count={int(structure_debug.get('add_count', 0))}, "
        f"drop_ratio={float(structure_debug.get('drop_ratio', 0.0)):.6f}, "
        f"keep_ratio={float(structure_debug.get('keep_ratio', 0.0)):.6f}, "
        f"delta_norm={float(structure_debug.get('delta_norm', 0.0)):.6f}, "
        f"move_ratio={float(structure_debug.get('move_ratio', 0.0)):.6f}, "
        f"preserve_ratio={float(structure_debug.get('preserve_ratio', 0.0)):.6f}"
    )

    writer.write(
        f"VoxelOperationStats step={step + 1}/{num_steps}: "
        f"before_occ={int(structure_debug.get('before_occupied_voxel_count', 0))}, "
        f"after_occ={int(structure_debug.get('after_occupied_voxel_count', 0))}, "
        f"occ_delta={int(structure_debug.get('occupied_voxel_delta', 0))}, "
        f"delete_voxels={int(structure_debug.get('delete_target_voxel_count', 0))}, "
        f"delete_removed_points={int(structure_debug.get('delete_removed_point_count', 0))}, "
        f"add_voxels={int(structure_debug.get('add_target_voxel_count', 0))}, "
        f"add_points={int(structure_debug.get('add_actual_point_count', 0))}, "
        f"move_source_voxels={int(structure_debug.get('move_source_voxel_count', 0))}, "
        f"move_target_voxels={int(structure_debug.get('move_target_voxel_count', 0))}"
    )


def log_point_edit_stats(writer, train_edit_stats, step, num_steps):
    if not train_edit_stats:
        return

    input_avg = float(
        train_edit_stats.get(
            "input_points_avg",
            train_edit_stats.get("input_points", 0),
        )
    )
    pre_output_avg = float(
        train_edit_stats.get(
            "pre_output_points_avg",
            train_edit_stats.get("pre_output_points", 0),
        )
    )
    output_avg = float(
        train_edit_stats.get(
            "output_points_avg",
            train_edit_stats.get("output_points", 0),
        )
    )

    writer.write(
        "PointEditStats "
        f"step={step + 1}/{num_steps}: "
        f"input_mean={input_avg:.3f}, "
        f"pre_output_mean={pre_output_avg:.3f}, "
        f"output_mean={output_avg:.3f}, "
        f"added_ratio={float(train_edit_stats.get('added_ratio_percent', 0.0)):.4f}%, "
        f"deleted_ratio={float(train_edit_stats.get('deleted_ratio_percent', 0.0)):.4f}%, "
        f"adjusted_ratio={float(train_edit_stats.get('adjusted_ratio_percent', 0.0)):.4f}%, "
        f"adjust_mean={float(train_edit_stats.get('adjust_mean', 0.0)):.6g}, "
        f"adjust_max={float(train_edit_stats.get('adjust_max', 0.0)):.6g}, "
        f"keep_mode={train_edit_stats.get('keep_mode', 'none')}"
    )