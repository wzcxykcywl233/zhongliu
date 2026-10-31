"""Same objective/optimizer operations, measured by phase with optional CPU evidence."""
import torch
import binding as B
from domain import domain_loss, schedule, gradient_norm


def execute(model, sample, label, step, arm, params, classifier, opt, disc_opt,
            head, tap, phases, audit_grads, evidence=False):
    P, T = B.P, B.BASE.T
    side = [p for n, p in model.named_parameters() if n.startswith(P.PREFIX)]
    enc = [p for n, p in model.named_parameters() if n.startswith(P.ENCODER_PREFIXES) and p.requires_grad]
    kind, maximum = P.ARMS[arm]
    T.seed_all(sample['draw'] + 47)
    opt.zero_grad(set_to_none=True)
    if disc_opt is not None:
        disc_opt.zero_grad(set_to_none=True)
    if tap:
        tap.reset(len(sample['video']))
    with phases.part('host_to_device'):
        video, queries = sample['video'][None].cuda(), sample['queries'].cuda()
        target = {k: label[k].cuda() for k in ('tracks', 'visibility')}
    with phases.part('forward'):
        out = model(video=video, queries=queries, iters=P.C.RULE['iterations'], is_train=True)
    with phases.part('loss_and_features'):
        task = T.S.training_loss(out, target)
        strength = schedule(step, maximum)
        adversarial = task.new_zeros(())
        probability = None
        diag = {}
        if head is not None:
            embedding = tap.embedding()
            adversarial, prob = domain_loss(head, embedding, sample['domain'], strength)
            probability = float(prob.item())
            diag = dict(feature_rms=float(embedding.detach().square().mean().sqrt()),
                        feature_channel_std=float(embedding.detach().std()),
                        pooled_feature=embedding.detach().cpu().flatten().tolist(),
                        domain_correct=(probability >= .5) == (sample['domain'] == 'lld_synthetic'))
    if audit_grads:
        if enc:
            with phases.part('task_gradient_audit'):
                diag['encoder_task_grad_norm'] = gradient_norm(task, enc)
                if diag['encoder_task_grad_norm'] <= 0:
                    raise ValueError('Task cannot update encoder')
        if head is not None and strength > 0:
            with phases.part('adversarial_gradient_audit'):
                diag['adversarial_grad_norm'] = gradient_norm(adversarial, side if kind == 'sidecar' else enc)
                if diag['adversarial_grad_norm'] <= 0:
                    raise ValueError('GRL recipient has no gradient')
    with phases.part('backward'):
        loss = task + adversarial
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite objective')
        loss.backward()
    with phases.part('gradient_check_and_clip'):
        for group in (params, classifier):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in group):
                raise ValueError('Nonfinite gradient')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('Unexpected frozen gradient')
        sn = float(torch.nn.utils.clip_grad_norm_(params, P.RULE['student_clip']))
        dn = float(torch.nn.utils.clip_grad_norm_(classifier, P.RULE['classifier_clip'])) if classifier else 0.
    packet = None
    if evidence:
        # Not timed as training work; no GPU tensor persists across jobs.
        packet = dict(outputs=[v.detach().cpu().clone() for v in out[:3]],
                      gradients={n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None},
                      classifier_gradients={n: p.grad.detach().cpu().clone() for n, p in head.named_parameters() if p.grad is not None} if head else {})
    with phases.part('optimizer_update'):
        opt.step()
        if disc_opt is not None:
            disc_opt.step()
    if evidence:
        packet['student'] = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        packet['classifier'] = {n: v.detach().cpu().clone() for n, v in head.state_dict().items()} if head else {}
    row = dict(point=float(task.detach()), domain_loss=float(adversarial.detach()),
               grl_strength=strength, domain_probability=probability, student_grad_norm=sn,
               classifier_grad_norm=dn, **diag)
    if tap:
        tap.values.clear()
    return row, packet
