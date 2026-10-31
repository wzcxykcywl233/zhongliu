"""Gate first, then reuse the original queue/report semantics in a NEW output root."""
import argparse
import binding as B


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['plan','train','audit','infer','record','report','roster','test-gate'])
    p.add_argument('--phase', choices=['smoke','pilot','full'], default='pilot')
    p.add_argument('--arm', choices=['pretrained']+list(B.P.ARMS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', choices=['validation-10','test-38'], default='validation-10')
    p.add_argument('--folder'); p.add_argument('--require-inference', action='store_true')
    a = p.parse_args()
    root, previous, audit = [B.Path(x) for x in ('/study','/previous','/audit')]
    P = B.configure_queue(B.Path('/diagnostic'),B.Path('/baseline'),previous,audit)
    if a.command == 'plan': P.prepare(root,previous,audit); return
    P.checked(root,previous)
    if a.command == 'roster':
        P.C.save(root/'full-roster.json',dict(plan=P.C.sha(root/'plan.json'),arms=P.arms_for(root,'full')))
        return
    if a.command == 'test-gate': P.test_gate(root); return
    import instrumented_train as T
    import evaluation as E
    E.audit_training = T.audit_training
    if a.command in ('train','infer'):
        from diagnose import environment
        if environment() != P.C.read(B.Path('/diagnostic/environment.json')):
            raise ValueError('GPU/runtime differs from the verified diagnostic')
    if a.command == 'train':
        T.train(root,previous,B.Path('/weights'),audit,B.Path('/lld'),B.Path('/data'),a.phase,a.arm,a.seed)
    elif a.command == 'audit': T.audit_training(root,previous,a.phase,a.require_inference)
    elif a.command == 'infer': E.infer(root,previous,B.Path('/data'),B.Path('/weights'),a.phase,a.split,a.arm,a.seed)
    elif a.command == 'record': E.record(root,previous,a.folder)
    elif a.command == 'report': E.report(root,previous,a.phase,a.split)


if __name__ == '__main__': main()
