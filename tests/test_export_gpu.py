import importlib.util

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from conftest import ROOT

pytestmark = pytest.mark.gpu


def _upstream_evaluate():
    spec = importlib.util.spec_from_file_location("upstream_evaluate",
                                                  ROOT / "external" / "FashionMV" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exported_model_matches_encoder_under_upstream_loader(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from gpu_helpers import make_perturbed_run
    from procir.collators import IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS
    from procir_train.encoder import load_encoder
    from procir_train.export import export_merged
    from procir_train.prompts import doc_text, multiturn_query_text, process_visual
    from procir_train.train import load_checkpoint

    run = tmp_path / "run"
    cfg = make_perturbed_run(run)
    export_merged(run / "config.yaml", run / "ckpt" / "latest.pt", tmp_path / "export")

    device = torch.device("cuda")
    from transformers import AutoConfig, AutoTokenizer

    up_model, _, up_emb_id = _upstream_evaluate().setup_model(str(tmp_path / "export"), device)
    enc, proc = load_encoder(cfg, device)
    load_checkpoint(run / "ckpt" / "latest.pt", enc)
    enc.eval()
    base, _ = load_encoder(cfg, device)  # same architecture, no trained deltas
    base.eval()
    assert up_emb_id == enc.emb_token_id
    assert AutoTokenizer.from_pretrained(str(tmp_path / "export")).convert_tokens_to_ids("<emb_all>") \
        == enc.emb_token_id
    base_cfg = AutoConfig.from_pretrained(cfg.base_model)
    vocab = getattr(base_cfg, "text_config", base_cfg).vocab_size
    assert up_model.vlm.get_input_embeddings().weight.shape[0] == vocab  # no resize

    imgs = [Image.new("RGB", (240, 320), (i * 60, 90, 120)) for i in range(2)]
    px = (IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS)
    doc_in = [process_visual(proc, doc_text(proc, 2), imgs, *px)]
    q_in = [process_visual(proc, multiturn_query_text(proc, 2, "make it red"), imgs, *px)]
    with torch.no_grad():
        ours_d, base_d = enc.encode_single(doc_in)[0].float(), base.encode_single(doc_in)[0].float()
        theirs_d = up_model.forward_visual_batch(doc_in, device)[0].float()
        ours_q, base_q = enc.encode_multiturn(q_in)[1][0].float(), base.encode_multiturn(q_in)[1][0].float()
        theirs_q = up_model.forward_visual_batch_multiturn(q_in, device)[1][0].float()

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    assert cos(ours_d, theirs_d) > 0.999 and cos(ours_q, theirs_q) > 0.999
    # the deltas must matter, otherwise the checks above prove nothing; compare the shifts themselves
    assert (ours_d - base_d).norm() > 0.05 * base_d.norm() and (ours_q - base_q).norm() > 0.05 * base_q.norm()
    assert cos(ours_d - base_d, theirs_d - base_d) > 0.99 and cos(ours_q - base_q, theirs_q - base_q) > 0.99
