import argparse
import full_context as X


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['plan','train','audit','infer','record','report','test-gate'])
    p.add_argument('--arm',choices=['pretrained']+list(X.ARMS)); p.add_argument('--seed',type=int,choices=X.RULE['seeds'],default=0)
    p.add_argument('--split',choices=['validation-10','test-38'],default='validation-10'); p.add_argument('--folder')
    a=p.parse_args()
    root,previous,audit,baseline,evidence,stability=[X.Path(x) for x in ('/study','/previous','/audit','/baseline','/evidence','/stability')]
    if a.command=='plan': X.prepare(root,previous,audit,baseline,evidence,stability); return
    X.provenance(root,previous,audit,baseline,evidence,stability)
    if a.command=='test-gate': X.test_gate(root); print('PLAN TEST GATE VERIFIED; frozen validation roster',flush=True); return
    if a.command in ('train','infer'): X.H.check_environment(root,stability)
    import training as T
    import full_evaluation as E
    if a.command=='train': T.train(root,previous,X.Path('/weights'),audit,X.Path('/lld'),X.Path('/data'),a.arm,a.seed)
    elif a.command=='audit': T.audit_training(root,previous)
    elif a.command=='infer': E.infer(root,previous,X.Path('/data'),X.Path('/weights'),'full',a.split,a.arm,a.seed)
    elif a.command=='record': E.record(root,previous,a.folder)
    elif a.command=='report': E.report(root,previous,'full',a.split)


if __name__=='__main__': main()
