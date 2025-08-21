import argparse
import csv
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompts(num_prompts: int = 20, k_support: int = 4) -> List[Tuple[str, str, str]]:
    """Return list of (context, query, full) strings.
    We use a simple templated English pattern to elicit contextual effects without task labels.
    """
    bases = [
        ("oscar", "cat"), ("luna", "dog"), ("milo", "bird"), ("ivy", "tree"), ("atlas", "river"),
        ("nova", "galaxy"), ("ember", "fire"), ("zephyr", "wind"), ("terra", "earth"), ("sol", "sun")
    ]
    out = []
    for i in range(num_prompts):
        ctx_pairs = []
        for j in range(k_support):
            a, b = bases[(i + j) % len(bases)]
            ctx_pairs.append(f"{a} is to {b}.")
        context = " ".join(ctx_pairs)
        a, b = bases[(i + k_support) % len(bases)]
        query = f" {a} is to"
        full = context + query
        out.append((context, query, full))
    return out


def capture_block_pre_mlp(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int):
    """Run model and capture the last block's pre-MLP representation (input to ln_2) at the last token.
    We register a forward hook on ln_2 to read its input, which equals y = x + Attn(LN1(x)) just before MLP.
    Returns: logits [B, V], A_repr [B, d_model]
    """
    device = next(model.parameters()).device
    ln2_inputs = []

    def hook_ln2(module, inp, out):
        # inp is a tuple; we want the tensor input to LayerNorm: [B,T,D]
        ln2_inputs.append(inp[0].detach())

    # last block
    block = model.transformer.h[layer_idx]
    handle = block.ln_2.register_forward_hook(hook_ln2)
    try:
        out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = out.logits[:, -1, :]  # [B, V]
        assert len(ln2_inputs) > 0, "ln_2 hook did not capture input"
        y = ln2_inputs[-1][:, -1, :]  # [B, D]
        return logits.detach().cpu(), y.detach().cpu()
    finally:
        handle.remove()


@torch.no_grad()
def compute_delta_rank1(model, A_Cx: torch.Tensor, A_x: torch.Tensor, per_sample: bool = True, layer_idx: int = -1):
    """Compute a rank-1 delta for the last block's first MLP layer (c_fc).
    A_Cx, A_x: [B, D]
    If per_sample=True, compute a per-sample rank-1 dW and average across the batch.
    Returns dW with the same shape as c_fc.weight: [4D, D] for GPT-2 MLP.
    """
    block = model.transformer.h[layer_idx]
    c_fc = block.mlp.c_fc  # Conv1D
    W = c_fc.weight  # HF Conv1D typically stores [in_features, out_features]
    # We will construct dW to match W.shape exactly
    diff = (A_Cx - A_x)  # [B, D]
    # Build a crude proxy u in 4D space using c_proj^T @ avg lm_head direction
    c_proj = block.mlp.c_proj  # Conv1D(out=D, in=4D)
    lm_head_w = model.lm_head.weight  # [V, D]
    lm_dir = lm_head_w.mean(dim=0, keepdim=True).t()  # [D,1]
    # Use correct orientation so result lives in 4D space
    u_base = c_proj.weight @ lm_dir  # [4D,1]
    u_base = (u_base / (u_base.norm() + 1e-8)).squeeze(1)  # [4D]

    if per_sample:
        # Decide outer orientation based on W.shape
        if W.shape[0] < W.shape[1]:
            # W is [D, 4D] (in, out) as in HF Conv1D: use outer(v, u)
            dW_acc = torch.zeros_like(W, device=u_base.device, dtype=u_base.dtype)
            u1 = u_base.flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [4D]
            for i in range(diff.shape[0]):
                v_i = diff[i : i + 1, :]
                v_i = (v_i / (v_i.norm() + 1e-8)).squeeze(0).flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [D]
                dW_acc += torch.outer(v_i, u1)  # [D, 4D]
        else:
            # W is [4D, D] (out, in): use outer(u, v)
            dW_acc = torch.zeros_like(W, device=u_base.device, dtype=u_base.dtype)
            u1 = u_base.flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [4D]
            for i in range(diff.shape[0]):
                v_i = diff[i : i + 1, :]
                v_i = (v_i / (v_i.norm() + 1e-8)).squeeze(0).flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [D]
                dW_acc += torch.outer(u1, v_i)  # [4D, D]
        dW = dW_acc / max(diff.shape[0], 1)
    else:
        v = diff.mean(dim=0, keepdim=True)
        v = (v / (v.norm() + 1e-8)).squeeze(0)
        if W.shape[0] < W.shape[1]:
            dW = torch.outer(v.to(u_base.device, u_base.dtype), u_base)  # [D,4D]
        else:
            dW = torch.outer(u_base, v.to(u_base.device, u_base.dtype))  # [4D,D]
    return dW


@torch.no_grad()
def apply_delta_block(model, dW: torch.Tensor, layer_idx: int):
    block = model.transformer.h[layer_idx]
    c_fc = block.mlp.c_fc
    W = c_fc.weight
    dW_dev = dW.to(W.device, dtype=W.dtype)
    # Match shapes: some implementations store [D,4D] instead of [4D,D]
    if dW_dev.shape != W.shape:
        if dW_dev.t().shape == W.shape:
            dW_dev = dW_dev.t()
        else:
            raise RuntimeError(f"Shape mismatch: dW {dW_dev.shape} vs W {W.shape}")
    W += dW_dev


def texts_to_inputs(tokenizer, texts: List[str], device):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def main(model_name: str, num_prompts: int, k_support: int, out_csv: str, device: str,
         per_sample: bool, alpha_max: float, alpha_steps: int, layer: int, sweep_layers: bool):
    device = torch.device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    # Ensure padding is defined for causal LM batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.eval()

    header = ["layer", "idx", "mse_baseline", "mse_after_delta"] if sweep_layers else ["idx", "mse_baseline", "mse_after_delta"]
    rows = [tuple(header)]
    prompts = build_prompts(num_prompts=num_prompts, k_support=k_support)

    def run_for_layer(layer_idx: int):
        nonlocal rows
        deltas = []
        for i in range(num_prompts):
            context, query, full = prompts[i]
            # contextual logits and A(C,x)
            ids_full, mask_full = texts_to_inputs(tok, [full], device)
            logits_ctx, A_Cx = capture_block_pre_mlp(model, ids_full, mask_full, layer_idx)
            # query-only logits and A(x)
            ids_q, mask_q = texts_to_inputs(tok, [query], device)
            logits_q, A_x = capture_block_pre_mlp(model, ids_q, mask_q, layer_idx)
            # baseline mismatch
            mse_baseline = nn.functional.mse_loss(logits_ctx, logits_q).item()
            # Compute dW
            dW_base = compute_delta_rank1(model, A_Cx, A_x, per_sample=per_sample, layer_idx=layer_idx)
            # In-place two-stage alpha search on a single model copy
            model2_tmp = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            model2_tmp.eval()
            # Cache original weights for this layer
            W = model2_tmp.transformer.h[layer_idx].mlp.c_fc.weight
            W0 = W.detach().clone()

            def eval_alpha(a: float) -> float:
                # Restore and apply scaled delta
                with torch.no_grad():
                    W.copy_(W0)
                    apply_delta_block(model2_tmp, dW_base * a, layer_idx)
                logits_q_after, _ = capture_block_pre_mlp(model2_tmp, ids_q, mask_q, layer_idx)
                return nn.functional.mse_loss(logits_ctx, logits_q_after).item()

            # Coarse grid
            coarse_steps = min(max(alpha_steps, 5), 21)
            coarse_alphas = [alpha_max * t / max(coarse_steps - 1, 1) for t in range(coarse_steps)]
            best_alpha = 0.0
            best_mse = None
            for a in coarse_alphas:
                mse_a = eval_alpha(a)
                if (best_mse is None) or (mse_a < best_mse):
                    best_mse = mse_a
                    best_alpha = a

            # Fine grid around best (±20% of alpha_max)
            import numpy as np
            span = 0.2 * alpha_max
            a_lo = max(0.0, best_alpha - span)
            a_hi = min(alpha_max, best_alpha + span)
            fine_steps = 7
            fine_alphas = np.linspace(a_lo, a_hi, fine_steps).tolist()
            for a in fine_alphas:
                mse_a = eval_alpha(a)
                if mse_a < best_mse:
                    best_mse = mse_a
                    best_alpha = a
            # Ensure weights restored
            with torch.no_grad():
                W.copy_(W0)
            mse_after = best_mse
            deltas.append(mse_baseline - mse_after)
            if sweep_layers:
                rows.append((layer_idx, i, mse_baseline, mse_after))
            else:
                rows.append((i, mse_baseline, mse_after))
            print(f"layer={layer_idx} idx={i}: baseline={mse_baseline:.6e} after_delta={mse_after:.6e}")
        # Summary for this layer
        import numpy as np
        deltas_np = np.array(deltas)
        improved = (deltas_np > 0).mean() * 100.0
        mean_drop = deltas_np.mean()
        median_drop = np.median(deltas_np)
        print(f"Summary layer {layer_idx}: %improved={improved:.1f} mean_drop={mean_drop:.4f} median_drop={median_drop:.4f}")

    if sweep_layers:
        n_layers = len(model.transformer.h)
        for L in range(n_layers):
            run_for_layer(L)
    else:
        run_for_layer(layer if layer is not None else -1)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="distilgpt2")
    p.add_argument("--num_prompts", type=int, default=20)
    p.add_argument("--k_support", type=int, default=4)
    p.add_argument("--out_csv", type=str, default="gpt_experiments/distilgpt2_delta_equivalence.csv")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--per_sample", action="store_true")
    p.add_argument("--alpha_max", type=float, default=2.0)
    p.add_argument("--alpha_steps", type=int, default=21)
    p.add_argument("--layer", type=int, default=None, help="Layer index to use (default: last)")
    p.add_argument("--sweep_layers", action="store_true", help="Sweep all layers and include layer column in CSV")
    args = p.parse_args()
    main(args.model, args.num_prompts, args.k_support, args.out_csv, args.device,
         args.per_sample, args.alpha_max, args.alpha_steps, args.layer, args.sweep_layers)
