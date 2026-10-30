import argparse
from protocol import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['plan','train','audit','infer','record','report','roster','test-gate'])
    p.add_argument('--phase',choices=['smoke','pilot','full'],default='pilot')
    p.add_argument('--arm',choices=['pretrained']+list(ARMS)); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--split',choices=['validation-10','test-38'],default='validation-10')
    p.add_argument('--folder'); p.add_argument('--require-inference',action='store_true'); a=p.parse_args()
    root=Path('/study'); previous=Path('/previous')
    if a.command=='plan': prepare(root,previous,Path('/audit')); return
    checked(root,previous)
    if a.command=='roster':
        arms=arms_for(root,'full')
        C.save(root/'full-roster.json',dict(plan=C.sha(root/'plan.json'),arms=arms))
        print('PLAN FULL ROSTER '+(','.join(arms) if arms else 'CLOSED'),flush=True); return
    if a.command=='test-gate': test_gate(root); print('PLAN TEST GATE VERIFIED',flush=True); return
    from train_domain import train,audit_training
    from evaluation import infer,record,report
    if a.command=='train': train(root,previous,Path('/weights'),Path('/audit'),Path('/lld'),Path('/data'),a.phase,a.arm,a.seed)
    elif a.command=='audit': audit_training(root,previous,a.phase,a.require_inference)
    elif a.command=='infer': infer(root,previous,Path('/data'),Path('/weights'),a.phase,a.split,a.arm,a.seed)
    elif a.command=='record': record(root,previous,a.folder)
    elif a.command=='report': report(root,previous,a.phase,a.split)


if __name__=='__main__': main()
