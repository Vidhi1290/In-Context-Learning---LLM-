import argparse
from typing import List, Tuple
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def capture_last_block_pre_mlp(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    ln2_inputs: List[torch.Tensor] = []

    def hook(module, inp, out):
        # inp is a tuple (hidden_states,)
        ln2_inputs.append(inp[0].detach())

    last_block = model.transformer.h[-1]
    handle = last_block.ln_2.register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]  # [B, V]
        assert len(ln2_inputs) > 0, "ln_2 hook did not capture input"
        y = ln2_inputs[-1][:, -1, :]  # [B, D]
        return logits.detach().cpu(), y.detach().cpu()
    finally:
        handle.remove()


@torch.no_grad()
def apply_delta_last_block(model, dW: torch.Tensor):
    last_block = model.transformer.h[-1]
    c_fc = last_block.mlp.c_fc
    W = c_fc.weight
    dW_dev = dW.to(W.device, dtype=W.dtype)
    # Support either [in, out] or [out, in]
    if dW_dev.shape == W.shape:
        W += dW_dev
    else:
        if W.shape[0] < W.shape[1] and dW_dev.shape[::-1] == W.shape:
            W += dW_dev.t()
        elif W.shape[0] > W.shape[1] and dW_dev.shape[::-1] == W.shape:
            W += dW_dev.t()
        else:
            raise RuntimeError(f"Shape mismatch: dW {dW_dev.shape} vs W {W.shape}")


@torch.no_grad()
def compute_delta_rank1(model, A_Cx: torch.Tensor, A_x: torch.Tensor, per_sample: bool = True):
    """Return dW matching c_fc.weight.shape exactly."""
    last_block = model.transformer.h[-1]
    c_fc = last_block.mlp.c_fc
    W = c_fc.weight
    diff = (A_Cx - A_x)  # [B, D]

    # Build u in 4D space via c_proj @ avg lm_head dir
    c_proj = last_block.mlp.c_proj  # Conv1D(out=D, in=4D)
    lm_head_w = model.lm_head.weight  # [V, D]
    lm_dir = lm_head_w.mean(dim=0, keepdim=True).t()  # [D,1]
    u_base = c_proj.weight @ lm_dir  # [4D,1]
    u_base = (u_base / (u_base.norm() + 1e-8)).squeeze(1)  # [4D]

    if per_sample:
        if W.shape[0] < W.shape[1]:
            # [in=D, out=4D]
            dW_acc = torch.zeros_like(W, device=u_base.device, dtype=u_base.dtype)
            u1 = u_base.flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [4D]
            for i in range(diff.shape[0]):
                v_i = diff[i : i + 1, :]
                v_i = (v_i / (v_i.norm() + 1e-8)).squeeze(0).flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [D]
                dW_acc += torch.outer(v_i, u1)  # [D,4D]
        else:
            # [out=4D, in=D]
            dW_acc = torch.zeros_like(W, device=u_base.device, dtype=u_base.dtype)
            u1 = u_base.flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [4D]
            for i in range(diff.shape[0]):
                v_i = diff[i : i + 1, :]
                v_i = (v_i / (v_i.norm() + 1e-8)).squeeze(0).flatten().contiguous().to(dW_acc.device, dW_acc.dtype)  # [D]
                dW_acc += torch.outer(u1, v_i)  # [4D,D]
        dW = dW_acc / max(diff.shape[0], 1)
    else:
        v = diff.mean(dim=0, keepdim=True)
        v = (v / (v.norm() + 1e-8)).squeeze(0)
        if W.shape[0] < W.shape[1]:
            dW = torch.outer(v.to(u_base.device, u_base.dtype), u_base)  # [D,4D]
        else:
            dW = torch.outer(u_base, v.to(u_base.device, u_base.dtype))  # [4D,D]
    return dW


def texts_to_inputs(tokenizer, texts: List[str], device):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def line_search_alpha(model_name: str, device, dW_base: torch.Tensor, tok, query: str,
                      target_logits: torch.Tensor, alpha_max: float, alpha_steps: int) -> Tuple[float, float]:
    best_mse = None
    best_alpha = 0.0
    ids_q, mask_q = texts_to_inputs(tok, [query], device)
    for t in range(alpha_steps):
        alpha = alpha_max * t / max(alpha_steps - 1, 1)
        model_tmp = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model_tmp.eval()
        apply_delta_last_block(model_tmp, dW_base * alpha)
        logits_q_after, _ = capture_last_block_pre_mlp(model_tmp, ids_q, mask_q)
        mse_after_tmp = nn.functional.mse_loss(target_logits, logits_q_after).item()
        if (best_mse is None) or (mse_after_tmp < best_mse):
            best_mse = mse_after_tmp
            best_alpha = alpha
    return best_alpha, float(best_mse)


@torch.no_grad()
def generate_text(model, tok, prompt: str, device, max_new_tokens: int = 40) -> str:
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    out_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out_ids[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--alpha_max", type=float, default=2.0)
    p.add_argument("--alpha_steps", type=int, default=21)
    p.add_argument("--context", type=str, default=None, help="Support context text")
    p.add_argument("--query", type=str, default=None, help="Query text")
    p.add_argument("--max_new_tokens", type=int, default=40)
    args = p.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    if getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.eval()

    # Get inputs interactively if not passed
    context = args.context
    query = args.query
    if context is None:
        print("Enter context/support text (end with Ctrl-D):\n>")
        import sys
        context = sys.stdin.read().strip()
    if query is None:
        query = input("\nEnter query text> ").strip()

    full = context + "\n" + query

    # Compute logits and pre-MLP activations
    ids_full, mask_full = texts_to_inputs(tok, [full], device)
    logits_ctx, A_Cx = capture_last_block_pre_mlp(model, ids_full, mask_full)

    ids_q, mask_q = texts_to_inputs(tok, [query], device)
    logits_q, A_x = capture_last_block_pre_mlp(model, ids_q, mask_q)

    mse_baseline = nn.functional.mse_loss(logits_ctx, logits_q).item()

    # Compute dW and alpha
    dW_base = compute_delta_rank1(model, A_Cx, A_x, per_sample=True)
    best_alpha, best_mse = line_search_alpha(
        args.model, device, dW_base, tok, query, logits_ctx, args.alpha_max, args.alpha_steps
    )

    print(f"Baseline MSE: {mse_baseline:.6e}")
    print(f"Best alpha: {best_alpha:.3f} | After-ΔW MSE: {best_mse:.6e}")

    # Apply best ΔW to fresh model and generate before/after
    model_after = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model_after.eval()
    apply_delta_last_block(model_after, dW_base * best_alpha)

    gen_before = generate_text(model, tok, query, device, max_new_tokens=args.max_new_tokens)
    gen_after = generate_text(model_after, tok, query, device, max_new_tokens=args.max_new_tokens)

    print("\n=== Query ===\n" + query)
    print("\n=== Generation (before ΔW) ===\n" + gen_before)
    print("\n=== Generation (after ΔW) ===\n" + gen_after)


if __name__ == "__main__":
    main()
